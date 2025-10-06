# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import copy

from .zero_head import ZeroHead, ZeroHeadHarmonious


def build_head(cfg):

    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop('name')
    if name == 'ZeroHead':
        return ZeroHead(**head_cfg)
    else:
        raise NotImplementedError



def build_head_harmonious(cfg):

    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop('name')
    if name == 'ZeroHead':
        return ZeroHeadHarmonious(**head_cfg)
    else:
        raise NotImplementedError
