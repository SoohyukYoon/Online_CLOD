damo_yolo_s_block_list = [
    # Backbone
    ['backbone.block_list.0'],
    ['backbone.block_list.1'],
    ['backbone.block_list.2'],
    ['backbone.block_list.3'],
    ['backbone.block_list.4'],
    ['backbone.block_list.5'],
    
    # Neck
    # FPN
    ['neck.merge_5'], 
    ['neck.merge_4'], 
    ['neck.merge_3'],
    # PAN
    ['neck.bu_conv13'], 
    ['neck.bu_conv24'], 
    ['neck.merge_7'], 
    ['neck.bu_conv57'], 
    ['neck.merge_6'], 
    ['neck.bu_conv46'], 
    ['neck.bu_conv76'],
    
    # Head
    # ['head.gfl_reg'],
    # ['head.gfl_cls']
    ['head.gfl_reg','head.gfl_cls']
]

damo_yolo_s_block_list2 = [
    # Backbone
    ['backbone.block_list.0'],
    ['backbone.block_list.1'],
    ['backbone.block_list.2'],
    ['backbone.block_list.3'],
    ['backbone.block_list.4'],
    ['backbone.block_list.5'],
    
    # Neck
    # FPN
    ['neck.merge_5'], 
    ['neck.merge_4'], 
    ['neck.merge_3'],
    # PAN
    ['neck.bu_conv13', 
    'neck.bu_conv24', 
    'neck.merge_7', 
    'neck.bu_conv57', 
    'neck.merge_6', 
    'neck.bu_conv46', 
    'neck.bu_conv76'],
    
    # Head
    # ['head.gfl_reg'],
    # ['head.gfl_cls']
    ['head.gfl_reg','head.gfl_cls']
]

damo_yolo_s_block_list3 = [
    # Backbone
    ['backbone.block_list.0'],
    ['backbone.block_list.1'],
    ['backbone.block_list.2'],
    ['backbone.block_list.3'],
    ['backbone.block_list.4'],
    ['backbone.block_list.5'],
    
    # Neck
    # FPN
    ['neck.merge_5'], 
    ['neck.merge_4'], 
    ['neck.merge_3'],
    # PAN
    ['neck.bu_conv13', 
    'neck.bu_conv24',], 
    ['neck.merge_7'], 
    ['neck.bu_conv57'], 
    ['neck.merge_6'], 
    ['neck.bu_conv46', 
    'neck.bu_conv76'],
    
    # Head
    # ['head.gfl_reg'],
    # ['head.gfl_cls']
    ['head.gfl_reg','head.gfl_cls']
]

damo_yolo_s_block_list4 = [
    # Backbone
    ['backbone.block_list.0'],
    ['backbone.block_list.1'],
    ['backbone.block_list.2'],
    ['backbone.block_list.3'],
    ['backbone.block_list.4'],
    ['backbone.block_list.5'],
    
    # Neck
    # FPN
    ['neck.bu_conv13'],
    ['neck.merge_3'],
    ['neck.bu_conv24'],
    ['neck.merge_4'],
    ['neck.merge_5'],
    ['neck.bu_conv57'],
    ['neck.merge_7'],
    ['neck.merge_6'],
    
    # Head
    # ['head.gfl_reg'],
    # ['head.gfl_cls']
    ['head.gfl_reg','head.gfl_cls']
]

damo_yolo_s_block_list5 = [
    # Backbone
    ['backbone.block_list.0'],
    ['backbone.block_list.1'],
    ['backbone.block_list.2'],
    ['backbone.block_list.3'],
    ['backbone.block_list.4'],
    ['backbone.block_list.5'],
    
    # Neck
    # FPN
    ['neck.bu_conv13', 'neck.merge_3'],
    ['neck.bu_conv24','neck.merge_4'],
    ['neck.merge_5'],
    ['neck.bu_conv57','neck.merge_7'],
    ['neck.bu_conv46','neck.bu_conv76','neck.merge_6'],
    
    # Head
    # ['head.gfl_reg'],
    # ['head.gfl_cls']
    ['head.gfl_reg','head.gfl_cls']
]

MODEL_BLOCK_DICT = {
    'damo': damo_yolo_s_block_list,
    'damo2': damo_yolo_s_block_list2,
    'damo3': damo_yolo_s_block_list3,
    'damo4': damo_yolo_s_block_list4,
    'damo5': damo_yolo_s_block_list5,
}

def get_blockwise_flops(flops_dict, model_name, method=None):
    if model_name not in MODEL_BLOCK_DICT:
        raise KeyError(f"Model '{model_name}' not found in MODEL_BLOCK_DICT. Please add its block list.")
        
    block_prefix_list = MODEL_BLOCK_DICT[model_name]
    
    forward_flops = []
    backward_flops = []
    G_forward_flops = []
    G_backward_flops = []
    F_forward_flops = []
    F_backward_flops = []
    
    for block_prefixes in block_prefix_list:
        block_layers = []
        for prefix in block_prefixes:
            block_layers.extend([layer_name for layer_name in flops_dict if layer_name.startswith(prefix)])

        forward_flops.append(sum([flops_dict[layer]['forward_flops'] / 10e9 for layer in block_layers]))
        backward_flops.append(sum([flops_dict[layer]['backward_flops'] / 10e9 for layer in block_layers]))
        
        g_f, g_b, f_f, f_b = 0, 0, 0, 0
        
        if method == "remind":
            # REMIND: Backbone의 초기 절반(stage 0-2)만 G로, 나머지는 모두 F로 취급
            early_backbone_prefixes = ['backbone.block_list.0', 'backbone.block_list.1', 'backbone.block_list.2']
            for layer in block_layers:
                # 레이어가 초기 backbone 프리픽스 중 하나로 시작하는지 확인
                is_general = any(layer.startswith(p) for p in early_backbone_prefixes)
                if is_general:
                    g_f += flops_dict[layer]['forward_flops'] / 10e9
                    g_b += flops_dict[layer]['backward_flops'] / 10e9
                else:
                    f_f += flops_dict[layer]['forward_flops'] / 10e9
                    f_b += flops_dict[layer]['backward_flops'] / 10e9

        elif method == "memo":
            # MEMO: Backbone 전체를 G로, Neck과 Head를 F로 취급
            for layer in block_layers:
                if layer.startswith('backbone'):
                    g_f += flops_dict[layer]['forward_flops'] / 10e9
                    g_b += flops_dict[layer]['backward_flops'] / 10e9
                else:
                    f_f += flops_dict[layer]['forward_flops'] / 10e9
                    f_b += flops_dict[layer]['backward_flops'] / 10e9
     
        if method is not None:
            G_forward_flops.append(g_f)
            G_backward_flops.append(g_b)
            F_forward_flops.append(f_f)
            F_backward_flops.append(f_b)

    return forward_flops, backward_flops, G_forward_flops, G_backward_flops, F_forward_flops, F_backward_flops

# def get_blockwise_flops(flops_dict, model_name, method=None):

#     block_list = MODEL_BLOCK_DICT[model_name]
    
#     forward_flops = []
#     backward_flops = []
#     G_forward_flops = []
#     G_backward_flops = []
#     F_forward_flops = []
#     F_backward_flops = []
#     G_forward, G_backward, F_forward, F_backward = [], [], [], []
    
#     for block in block_list:
#         forward_flops.append(sum([flops_dict[layer]['forward_flops']/10e9 for layer in block]))
#         backward_flops.append(sum([flops_dict[layer]['backward_flops']/10e9 for layer in block]))
    
#         if method=="remind":
#             for layer in block:
#                 if "model_G" in layer:
#                     G_forward.append(flops_dict[layer]['forward_flops']/10e9)
#                     G_backward.append(flops_dict[layer]['backward_flops']/10e9)
#                 else:
#                     F_forward.append(flops_dict[layer]['forward_flops']/10e9)
#                     F_backward.append(flops_dict[layer]['backward_flops']/10e9)
                    
#             G_forward_flops.append(sum(G_forward))
#             G_backward_flops.append(sum(G_backward))
#             F_forward_flops.append(sum(F_forward))
#             F_backward_flops.append(sum(F_backward))
            
#         elif method=="memo":
#             for layer in block:
#                 if "backbone" in layer:
#                     G_forward.append(flops_dict[layer]['forward_flops']/10e9)
#                     G_backward.append(flops_dict[layer]['backward_flops']/10e9)
#                 elif "AdaptiveExtractors" in layer or "fc" in layer:
#                     F_forward.append(flops_dict[layer]['forward_flops']/10e9)
#                     F_backward.append(flops_dict[layer]['backward_flops']/10e9)
  
#             G_forward_flops.append(sum(G_forward))
#             G_backward_flops.append(sum(G_backward))
#             F_forward_flops.append(sum(F_forward))
#             F_backward_flops.append(sum(F_backward))


#     return forward_flops, backward_flops, G_forward_flops, G_backward_flops, F_forward_flops, F_backward_flops