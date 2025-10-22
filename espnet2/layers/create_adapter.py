"""Definition of the low-rank adaptation (LoRA) for large models.

References:
    1. LoRA: Low-Rank Adaptation of Large Language Models
       (https://arxiv.org/pdf/2106.09685.pdf)
    2. https://github.com/microsoft/LoRA.git
    3. https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora.py

"""

import torch
from typeguard import typechecked

from espnet2.layers.create_adapter_fn import create_houlsby_adapter, \
                                             create_lora_adapter, \
                                             create_vera_adapter, \
                                             create_dict_lora_adapter, \
                                             create_dict_lora4lanfusion_adapter, \
                                             create_dict_lora4velora_adapter, \
                                             create_dict_lora4fastervelora_adapter, \
                                             create_dict_lora4cat_adapter, \
                                             create_dict_lora4ecam_adapter, \
                                             create_dict_lora4pcam_adapter, \
                                             create_dict_lora4mole_adapter, \
                                             create_dict_lora4samd_adapter, \
                                             create_moslora_adapter, \
                                             create_melora_adapter, \
                                             create_lora_houslby_adapter

create_adapter_fn_table = {

    "dictlora": create_dict_lora_adapter, # create LoRA weight using ParameterDict
    "dictlora4lanfusion": create_dict_lora4lanfusion_adapter,
    "dictlora4velora": create_dict_lora4velora_adapter,
    "dictlora4fastervelora": create_dict_lora4fastervelora_adapter,
    "dictlora4cat": create_dict_lora4cat_adapter,
    "dictlora4ecam": create_dict_lora4ecam_adapter,
    "dictlora4pcam": create_dict_lora4pcam_adapter,
    "dictlora4mole": create_dict_lora4mole_adapter,
    "dictlora4samd": create_dict_lora4samd_adapter,
    "lora": create_lora_adapter,
    "vera": create_vera_adapter,

    "houlsby": create_houlsby_adapter,
    "moslora": create_moslora_adapter,
    "melora": create_melora_adapter,
    "lora_houlsby": create_lora_houslby_adapter,
    # "vera": ,
}


@typechecked
def create_adapter(
    model: torch.nn.Module,
    adapter: str,
    adapter_conf: dict,
):
    """Create adapter for the base model.


    Args:
        model (torch.nn.Module): Base model to be adapted.
        adapter_type (str): Name of adapter
        adapter_conf (dict): Configuration for the adapter
            e.g.  {"rank": 8, "alpha": 8, ...} for lora

    """
    assert adapter in create_adapter_fn_table, f"Adapter {adapter} is not supported."
    create_adapter_fn = create_adapter_fn_table[adapter]
    create_adapter_fn(model=model, **adapter_conf)
