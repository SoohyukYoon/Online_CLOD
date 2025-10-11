import logging

from methods.er_baseline import ER
# from methods.ewc import EWCpp
# from methods.mir import MIR
# from methods.clib import CLIB
# from methods.gss import GSS
# from methods.der import DER
from methods.sdp import *
from methods.erd import *
from methods.ld import LD

from methods.adaptive_freeze import AdaptiveFreeze
from methods.er_freq_adaptive import ERFreqAdaptive
from methods.baseline import BASELINE
from methods.baseline2 import BASELINE2
from methods.baseline2_balanced import BASELINE2Balanced
from methods.baseline2_frequency import BASELINE2Frequency
from methods.baseline2_freq_balanced import BASELINE2FreqBalanced
from methods.er_freq_balanced import ERFreqBalanced
from methods.er_frequency import ERFrequency
from methods.er_balanced import ERBalanced
from methods.er_freq_balanced2 import ERFreqBalanced2
from methods.harmonious import Harmonious

from methods.finetune import FINETUNE
from methods.sdp import SDP
from methods.sdp_only import SDPOnly
from methods.lwf_logit import LWF_Logit
from methods.lwf_feature_extraction import LWF_Feature
# from methods.abr import ABR
from methods.er_pseudo import ERPseudo
from methods.er_freq_balanced_pseudo import ERFreqBalancedPseudo

from methods.baseline2_freq_balanced_pseudo_tia import BASELINEFreqBalancedPseudoGRAM
from methods.baseline2_freq_balanced_pseudo_tia_gram import BASELINEFreqBalancedPseudoGRAM2

from methods.er_selection_balanced import SampleSelection
from methods.er_selection import SampleSelectionBase

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
    elif args.mode == "er_selection_balanced":
        method = SampleSelection(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )  
    elif args.mode == "er_selection":
        method = SampleSelectionBase(
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
    elif args.mode == "baseline2_freq_balanced":
        method = BASELINE2FreqBalanced(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "er_freq_balanced":
        method = ERFreqBalanced(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "er_freq_balanced2":
        method = ERFreqBalanced2(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "er_balanced":
        method = ERBalanced(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "er_pseudo":
        method = ERPseudo(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "er_freq_balanced_pseudo":
        method = ERFreqBalancedPseudo(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "pseudo_ours":
        method = BASELINEFreqBalancedPseudoGRAM(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "pseudo_gram_ours":
        method = BASELINEFreqBalancedPseudoGRAM2(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "er_frequency":
        method = ERFrequency(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "er_frequency_adaptive":
        method = ERFreqAdaptive(
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
    elif args.mode == "ld":
        method = LD(
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
    elif args.mode == "harmonious":
        method = Harmonious(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "abr":
        method = ABR(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    else:
        raise NotImplementedError("Choose the args.mode in [er, gdumb, rm, bic, ewc++, mir, clib]")

    return method