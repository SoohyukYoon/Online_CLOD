import logging

from methods.er_baseline import ER
from methods.erd import ERD
from methods.ld import LD

from methods.adaptive_freeze import AdaptiveFreeze
from methods.er_freq_adaptive import ERFreqAdaptive
from methods.er_freq_balanced import ERFreqBalanced
from methods.er_frequency import ERFrequency
from methods.er_balanced import ERBalanced
from methods.harmonious import Harmonious

from methods.finetune import FINETUNE
from methods.sdp import SDP
from methods.lwf_logit import LWF_Logit
from methods.lwf_feature_extraction import LWF_Feature
# from methods.abr import ABR
from methods.er_pseudo import ERPseudo
from methods.er_freq_balanced_pseudo import ERFreqBalancedPseudo

from methods.baseline2_freq_balanced_pseudo_tia import BASELINEFreqBalancedPseudoGRAM
from methods.baseline2_freq_balanced_pseudo_tia_gram import BASELINEFreqBalancedPseudoGRAM2

from methods.er_selection_balanced import SampleSelectionBalanced
from methods.er_selection import SampleSelectionBase

logger = logging.getLogger()


def select_method(args, n_classes, device):
    kwargs = vars(args)
    if args.mode == "er":
        method = ER(
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "finetune":
        method = FINETUNE(
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "er_selection_balanced":
        method = SampleSelectionBalanced(
            device=device,
            n_classes=n_classes,
            **kwargs,
        )  
    elif args.mode == "er_selection":
        method = SampleSelectionBase(
            device=device,
            n_classes=n_classes,
            **kwargs,
        )  
    elif args.mode == "adaptive_freeze":
        method = AdaptiveFreeze(
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "er_freq_balanced":
        method = ERFreqBalanced(
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "er_balanced":
        method = ERBalanced(
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "er_pseudo":
        method = ERPseudo(
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "er_freq_balanced_pseudo":
        method = ERFreqBalancedPseudo(
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "pseudo_ours":
        method = BASELINEFreqBalancedPseudoGRAM(
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "pseudo_gram_ours":
        method = BASELINEFreqBalancedPseudoGRAM2(
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "er_frequency":
        method = ERFrequency(
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "er_frequency_adaptive":
        method = ERFreqAdaptive(
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "sdp":
        method = SDP(
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "ld":
        method = LD(
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "lwf_feature":
        method = LWF_Feature(
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "lwf_logit":
        method = LWF_Logit(
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "erd":
        method = ERD(
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "harmonious":
        method = Harmonious(
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    # elif args.mode == "abr":
    #     method = ABR(
    #         device=device,
    #         n_classes=n_classes,
    #         **kwargs,
    #     )
    else:
        raise NotImplementedError("Choose the args.mode in [er, gdumb, rm, bic, ewc++, mir, clib]")

    return method