from methods.er_baseline import ER
from methods.er_frequency import ERFrequency
import torch
from torch import nn
import numpy as np
from operator import attrgetter

class ERFreqAdaptive(ERFrequency):
    def __init__(self, n_classes, device, **kwargs):
        super().__init__(n_classes=n_classes, device=device, **kwargs)
        # Information based freezing
        self.unfreeze_rate = 0.0
        self.fisher_ema_ratio = 0.01
        self.fisher = [0.0 for _ in range(self.num_blocks)]
        self.last_grad_mean = 0.0
        # autograd_hacks.add_hooks(self.model)  # install once – cheap
        
        self.cumulative_fisher = []
        self.freeze_idx = []
        self.frozen = False

        self.cls_weight_decay = kwargs["cls_weight_decay"]

    def _layer_type(self, layer: nn.Module) -> str:
        return layer.__class__.__name__

    def prev_check(self, idx):
        result = True
        for i in range(idx):
            if i not in self.freeze_idx:
                result = False
                break
        return result

    def unfreeze_layers(self):
        self.frozen = False
        for name, param in self.model.named_parameters():
            # unfreeze = True
            # for aux_block_name in self.auxiliary_layers:
            #     if name.startswith(aux_block_name + '.'):
            #         param.requires_grad = False
            #         unfreeze = False
            #         break
            # if unfreeze:
            param.requires_grad = True

    def freeze_layers(self):
        if len(self.freeze_idx) > 0:
            self.frozen = True
        for i in self.freeze_idx:
            self.freeze_layer(i)

    def freeze_layer(self, block_index):
        # blcok(i)가 들어간 layer 모두 freeze
        block_name = self.block_names[block_index]
        # print("freeze", group_name)
        for subblock_name in block_name:
            for name, param in self.model.named_parameters():
                # if subblock_name in name:
                if name.startswith(subblock_name + '.'):
                    param.requires_grad = False

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        total_loss = 0.0
        if len(sample) > 0:
            self.memory.register_stream(sample)
        for i in range(iterations):
            self.model.train()
            data = self.memory.get_batch(batch_size, stream_batch_size)
            
            self.optimizer.zero_grad()

            loss, loss_item = self.model_forward(data)
            
            if self.train_count > 2:
                if self.unfreeze_rate < 1.0:
                    self.get_freeze_idx(loss)
                if np.random.rand() > self.unfreeze_rate:
                    self.freeze_layers()
            
            if len(self.freeze_idx) == self.num_blocks:
                continue
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                # autograd_hacks.compute_grad1(self.model)
                self.optimizer.step()
            
            # autograd_hacks.compute_grad1(self.model)
            
            if not self.frozen:
                self.calculate_fisher()
            
            # autograd_hacks.clear_backprops(self.model)
            
            self.update_schedule()

            total_loss += loss.item()
            
            if len(self.freeze_idx) == 0:    
                # forward와 backward가 full로 일어날 때
                self.total_flops += (len(data[1]) * (self.backward_flops))
            else:
                self.total_flops += (len(data[1]) * (self.get_backward_flops()))
            
            self.unfreeze_layers()
            self.freeze_idx = []
            self.train_count += 1
            
            
        return total_loss / iterations

    def get_backward_flops(self):
        backward_flops = self.backward_flops
        if self.frozen:
            for i in self.freeze_idx:
                backward_flops -= self.blockwise_backward_flops[i]
        return backward_flops

    @torch.no_grad()
    def calculate_fisher(self):
        block_fisher = [0.0 for _ in range(self.num_blocks)]
        # print('before fisher tflops', self.total_flops)
        for i, block_name in enumerate(self.block_names[:-1]):
            for subblock_name in block_name:
                get_attr = attrgetter(subblock_name)
                block = get_attr(self.model)
                block_grad = []
                block_input = []
                for n, p in block.named_parameters():
                    if p.requires_grad is True and p.grad is not None:
                        if not p.grad.isnan().any():
                            block_fisher[i] += (p.grad.clone().detach().clamp(-1, 1) ** 2).sum().item()
                            if self.unfreeze_rate < 1:
                                self.total_flops += len(p.grad.clone().detach().flatten())*2 / 10e9
        # print('after fisher tflops', self.total_flops)
        for i in range(self.num_blocks):
            if i not in self.freeze_idx or not self.frozen:
                self.fisher[i] += self.fisher_ema_ratio * (block_fisher[i] - self.fisher[i])
        # print(group_fisher)
        self.total_fisher = sum(self.fisher)
        self.cumulative_fisher = [sum(self.fisher[0:i+1]) for i in range(self.num_blocks)]
        # print(self.fisher, self.layerwise_backward_flops)

    def get_flops_parameter(self):
        super().get_flops_parameter()
        self.cumulative_backward_flops = [sum(self.blockwise_backward_flops[0:i+1]) for i in range(self.num_blocks)]
        print("self.cumulative_backward_flops")
        print(self.cumulative_backward_flops)

    @torch.no_grad()
    def get_freeze_idx(self, loss):
        ## FIXME: set param list
        # grad = self.get_grad(loss, [p for p in self.model.model[22].parameters()] + [p for p in self.model.model[30].parameters()])
        grad = self.get_grad(loss, [p for p in self.model.head.gfl_cls.parameters()]
                                    + [p for p in self.model.head.gfl_reg.parameters()]
                            )
                                #  + [p for p in self.model.head.gfl_cls[1].parameters()]
                                #  + [p for p in self.model.head.gfl_cls[2].parameters()])
        last_grad = (grad ** 2 ).sum().item()
        # print('before grad tflops', self.total_flops)
        if self.unfreeze_rate < 1:
            self.total_flops += len(grad.clone().detach().flatten())*2/10e9
        # print('after grad tflops', self.total_flops)
        batch_freeze_score = last_grad/(self.last_grad_mean+1e-10)
        self.last_grad_mean += self.fisher_ema_ratio * (last_grad - self.last_grad_mean)
        freeze_score = []
        freeze_score.append(1)
        # if 'noinitial' in self.note:
        freeze_score.append(1)
        total_model_flops = self.total_model_flops - self.blockwise_forward_flops[0] - self.blockwise_backward_flops[0]
        cumulative_backward_flops = [sum(self.blockwise_backward_flops[1:i+1]) for i in range(1, self.num_blocks)]
        total_fisher = sum(self.fisher[1:])
        cumulative_fisher = [sum(self.fisher[1:i+1]) for i in range(1, self.num_blocks)]
        
        for i in range(self.num_blocks-1):
            freeze_score.append(total_model_flops / (total_model_flops - cumulative_backward_flops[i]) * (
                        total_fisher - cumulative_fisher[i]) / (total_fisher + 1e-10))
        max_score = max(freeze_score)
        modified_score = []
        modified_score.append(batch_freeze_score)
        modified_score.append(batch_freeze_score)
        for i in range(self.num_blocks -1):
            modified_score.append(batch_freeze_score*(total_fisher - cumulative_fisher[i])/(total_fisher + 1e-10) + cumulative_backward_flops[i]/total_model_flops * max_score)
        optimal_freeze = np.argmax(modified_score)

        
        print(f'I/C: {freeze_score} \n BI/C: {modified_score} \n Grad_Magnitude: {batch_freeze_score}')
        print(f'Iter: {self.train_count} Freeze: {optimal_freeze}')
        self.freeze_idx = list(range(self.num_blocks))[0:optimal_freeze]

    @torch.no_grad()
    def get_grad(self, loss: torch.Tensor, param_list):
        # autograd will return *None* when the param is unused
        grads = torch.autograd.grad(
            loss,
            param_list,
            retain_graph=True,
            allow_unused=True,
        )

        flat = []
        for g, p in zip(grads, param_list):
            flat.append(
                (torch.zeros_like(p) if g is None else g)
                .contiguous()
                .view(-1)
            )
        grad_vec = torch.cat(flat)
        torch.cuda.empty_cache()
        return grad_vec
