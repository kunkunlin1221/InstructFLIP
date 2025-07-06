from lavis import registry
from lavis.models import load_preprocess as _load_preprocess
from omegaconf import OmegaConf


def load_preprocess(name, model_type):
    model_cls = registry.get_model_class(name)
    cfg = OmegaConf.load(model_cls.default_config_path(model_type))
    vis_processors, txt_processors = _load_preprocess(cfg.preprocess)
    return vis_processors, txt_processors


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
