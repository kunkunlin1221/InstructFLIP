from pytorch_warmup import ExponentialWarmup, LinearWarmup, RAdamWarmup, UntunedExponentialWarmup, UntunedLinearWarmup
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, OneCycleLR

from .cdcn import MMCDCN, CDCNpp
from .cfpl import CFPL
from .flip import FLIPMCL, FLIPV
from .instruct_flip import (InstructFLIP, InstructFLIP_VE, InstructFLIP_VE_CB, MultiClassifier, InstructFLIPnoLLM,
                            InstructFLIP_VE_CB_SB)
from .simple import SimpleClassifier
from .ssdg import SSDG
from .safas import SAFAS
from .vit import ViT

MODELs = {
    "SimpleClassifier": SimpleClassifier,
    "ViT": ViT,
    "CDCNpp": CDCNpp,
    "MMCDCN": MMCDCN,
    "FLIPV": FLIPV,
    "FLIPMCL": FLIPMCL,
    "CFPL": CFPL,
    "InstructFLIP": InstructFLIP,
    "InstructFLIP_VE": InstructFLIP_VE,
    "InstructFLIP_VE_CB": InstructFLIP_VE_CB,
    "InstructFLIP_VE_CB_SB": InstructFLIP_VE_CB_SB,
    "SSDG": SSDG,
    "SAFAS": SAFAS,
    'MultiClassifier': MultiClassifier,
    'InstructFLIPnoLLM': InstructFLIPnoLLM,
}

OPTIMs = {
    "SGD": SGD,
    "Adam": Adam,
    "AdamW": AdamW,
}

LRSCHEDULERs = {
    "OneCycleLR": OneCycleLR,
    "CosineAnnealingLR": CosineAnnealingLR,
    "MultiStepLR": MultiStepLR,
}


def build_model(cls_name, **kwargs):
    model_cls = MODELs.get(cls_name)
    if model_cls is None:
        raise ValueError(f"Unknown model: {cls_name}")
    return model_cls(**kwargs)


def build_optimizer(params, cls_name, **kwargs):
    optim_cls = OPTIMs.get(cls_name)
    if optim_cls is None:
        raise ValueError(f"Unknown optimizer: {cls_name}")
    return optim_cls(params=params, **kwargs)


def build_lr_scheduler(optimizer, cls_name, **kwargs):
    lr_scheduler_cls = LRSCHEDULERs.get(cls_name)
    if lr_scheduler_cls is None:
        raise ValueError(f"Unknown lr_scheduler: {cls_name}")
    return lr_scheduler_cls(optimizer=optimizer, **kwargs)


WARMUPs = {
    "RAdamWarmup": RAdamWarmup,
    "UntunedLinearWarmup": UntunedLinearWarmup,
    "UntunedExponentialWarmup": UntunedExponentialWarmup,
    "ExponentialWarmup": ExponentialWarmup,
    "LinearWarmup": LinearWarmup,
}


def build_warmup_scheduler(cls_name, optimizer, **kwargs):
    return WARMUPs[cls_name](optimizer, **kwargs)
