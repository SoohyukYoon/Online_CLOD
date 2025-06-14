import logging

from methods.er_baseline import ER
# from methods.ewc import EWCpp
# from methods.mir import MIR
# from methods.clib import CLIB
# from methods.gss import GSS
# from methods.der import DER
from methods.sdp import *
from methods.erd import *
from methods.ours import Ours
from methods.ours_min import OursMin
from methods.adaptive_freeze import AdaptiveFreeze

from methods.baseline import BASELINE
from methods.baseline2 import BASELINE2
from methods.baseline2_balanced import BASELINE2Balanced
from methods.baseline2_frequency import BASELINE2Frequency

from methods.finetune import FINETUNE
from methods.sdp import SDP
from methods.sdp_only import SDPOnly
from methods.lwf_logit import LWF_Logit
from methods.lwf_feature_extraction import LWF_Feature

logger = logging.getLogger()


def select_method(args, criterion, n_classes, device):
    kwargs = vars(args)
    if args.mode == "er":
        method = ER(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "baseline":
        method = BASELINE(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "finetune":
        method = FINETUNE(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "ours":
        method = Ours(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "ours_min":
        method = OursMin(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )  
    elif args.mode == "adaptive_freeze":
        method = AdaptiveFreeze(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "baseline2":
        method = BASELINE2(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "baseline2_balanced":
        method = BASELINE2Balanced(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "baseline2_frequency":
        method = BASELINE2Frequency(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "sdp":
        method = SDP(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "sdp_only":
        method = SDPOnly(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "lwf_feature":
        method = LWF_Feature(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "lwf_logit":
        method = LWF_Logit(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "erd":
        method = ERD(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    else:
        raise NotImplementedError("Choose the args.mode in [er, gdumb, rm, bic, ewc++, mir, clib]")

    return method