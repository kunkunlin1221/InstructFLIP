from typing import Any, Dict

import torch.nn as nn


class BaseModelInterface(nn.Module):

    '''
    ABC class to define the interface of a trainer with lightning module.
    Function are called in the lightningmodel(../lightningmodule.py)
    '''

    def _build_model(self, *args, **kwargs):
        return NotImplementedError

    def lazy_init(self, gpu_id):
        return NotImplementedError

    def forward_test(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return NotImplementedError

    def forward_train(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return NotImplementedError

    def show_detail(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        epoch: int,
        mode: str = 'train',
        logger=None,
        gpu_id: int = 0,
        **kwargs
    ):
        return NotImplementedError
