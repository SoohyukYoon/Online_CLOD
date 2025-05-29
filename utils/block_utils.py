
yolov9_s_block_list = [
    #backbone
    ['model.0'],
    ['model.1'],
    ['model.2'],
    ['model.3'],
    ['model.4'],
    ['model.5'],
    ['model.6'],
    ['model.7'],
    ['model.8'],
    #neck
    ['model.9', 'model.23'],
    ['model.10', 'model.11', 'model.12', 'model.24', 'model.25', 'model.26'],
    ['model.13', 'model.14', 'model.15', 'model.27', 'model.28', 'model.29'],
    ['model.16', 'model.17', 'model.18'],
    ['model.19', 'model.20', 'model.21'],
    ['model.30', 'model.22'] # detection head
]


MODEL_BLOCK_DICT = {
    'yolov9-s':yolov9_s_block_list,
}

def get_blockwise_flops(flops_dict, model_name, method=None):

    block_list = MODEL_BLOCK_DICT[model_name]
    
    forward_flops = []
    backward_flops = []
    G_forward_flops = []
    G_backward_flops = []
    F_forward_flops = []
    F_backward_flops = []
    G_forward, G_backward, F_forward, F_backward = [], [], [], []
    
    for block in block_list:
        forward_flops.append(sum([flops_dict[layer]['forward_flops']/10e9 for layer in block]))
        backward_flops.append(sum([flops_dict[layer]['backward_flops']/10e9 for layer in block]))
    
        if method=="remind":
            for layer in block:
                if "model_G" in layer:
                    G_forward.append(flops_dict[layer]['forward_flops']/10e9)
                    G_backward.append(flops_dict[layer]['backward_flops']/10e9)
                else:
                    F_forward.append(flops_dict[layer]['forward_flops']/10e9)
                    F_backward.append(flops_dict[layer]['backward_flops']/10e9)
                    
            G_forward_flops.append(sum(G_forward))
            G_backward_flops.append(sum(G_backward))
            F_forward_flops.append(sum(F_forward))
            F_backward_flops.append(sum(F_backward))
            
        elif method=="memo":
            for layer in block:
                if "backbone" in layer:
                    G_forward.append(flops_dict[layer]['forward_flops']/10e9)
                    G_backward.append(flops_dict[layer]['backward_flops']/10e9)
                elif "AdaptiveExtractors" in layer or "fc" in layer:
                    F_forward.append(flops_dict[layer]['forward_flops']/10e9)
                    F_backward.append(flops_dict[layer]['backward_flops']/10e9)
  
            G_forward_flops.append(sum(G_forward))
            G_backward_flops.append(sum(G_backward))
            F_forward_flops.append(sum(F_forward))
            F_backward_flops.append(sum(F_backward))


    return forward_flops, backward_flops, G_forward_flops, G_backward_flops, F_forward_flops, F_backward_flops