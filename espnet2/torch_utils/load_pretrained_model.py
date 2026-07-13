import logging
from typing import Any, Dict, Union

import torch
import torch.nn
import torch.optim


def filter_state_dict(
    dst_state: Dict[str, Union[float, torch.Tensor]],
    src_state: Dict[str, Union[float, torch.Tensor]],
):
    """Filter name, size mismatch instances between dicts.

    Args:
        dst_state: reference state dict for filtering
        src_state: target state dict for filtering

    """
    match_state = {}
    for key, value in src_state.items():
        if key in dst_state and (dst_state[key].size() == src_state[key].size()):
            match_state[key] = value
        else:
            if key not in dst_state:
                logging.warning(
                    f"Filter out {key} from pretrained dict"
                    + " because of name not found in target dict"
                )
            else:
                logging.warning(
                    f"Filter out {key} from pretrained dict"
                    + " because of size mismatch"
                    + f"({dst_state[key].size()}-{src_state[key].size()})"
                )
    return match_state


def load_pretrained_model(
    init_param: str,
    model: torch.nn.Module,
    ignore_init_mismatch: bool,
    id,
    map_location: str = "cpu",
    # following args (including "id") only sever for E-CAM
    adapter_type=None,
    num_models=1,
    src_experts_key_name=None,
    uniform_soup=None,
):
    """Load a model state and set it to the model.

    Args:
        init_param: <file_path>:<src_key>:<dst_key>:<exclude_Keys>

    Examples:
        >>> load_pretrained_model("somewhere/model.pth", model)
        >>> load_pretrained_model("somewhere/model.pth:decoder:decoder", model)
        >>> load_pretrained_model("somewhere/model.pth:decoder:decoder:", model)
        >>> load_pretrained_model(
        ...     "somewhere/model.pth:decoder:decoder:decoder.embed", model
        ... )
        >>> load_pretrained_model("somewhere/decoder.pth::decoder", model)
    """
    sps = init_param.split(":", 4)
    if len(sps) == 4:
        path, src_key, dst_key, excludes = sps
    elif len(sps) == 3:
        path, src_key, dst_key = sps
        excludes = None
    elif len(sps) == 2:
        path, src_key = sps
        dst_key, excludes = None, None
    else:
        (path,) = sps
        src_key, dst_key, excludes = None, None, None
    if src_key == "":
        src_key = None
    if dst_key == "":
        dst_key = None

    if dst_key is None:
        obj = model
    else:

        def get_attr(obj: Any, key: str):
            """Get an nested attribute.

            >>> class A(torch.nn.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.linear = torch.nn.Linear(10, 10)
            >>> a = A()
            >>> assert A.linear.weight is get_attr(A, 'linear.weight')

            """
            if key.strip() == "":
                return obj
            for k in key.split("."):
                obj = getattr(obj, k)
            return obj

        obj = get_attr(model, dst_key)

    src_state = torch.load(path, map_location=map_location)


    if adapter_type=="dictlora4ecam":
        if id != 0:
            # rename the src key
            src_state = {k.split("lora_A")[0]+"lora_A."+src_experts_key_name if "lora_A" in k else k : v for k, v in src_state.items()}
            src_state = {k.split("lora_B")[0]+"lora_B."+src_experts_key_name if "lora_B" in k else k : v for k, v in src_state.items()}

        if id == 1:
            uniform_soup = {k : v * (1./num_models) for k, v in src_state.items()}
            if id!=num_models:
                return uniform_soup
            else:
                src_state=uniform_soup
        elif id > 1:
            uniform_soup = {k : v * (1./num_models) + uniform_soup[k] for k, v in src_state.items()}
            if id!=num_models:
                return uniform_soup
            else:
                src_state=uniform_soup


    if excludes is not None:
        for e in excludes.split(","):
            src_state = {k: v for k, v in src_state.items() if not k.startswith(e)}

    if src_key is not None:
        src_state = {
            k[len(src_key) + 1 :]: v
            for k, v in src_state.items()
            if k.startswith(src_key)
        }

    dst_state = obj.state_dict()
    if ignore_init_mismatch:
        src_state = filter_state_dict(dst_state, src_state)
    dst_state.update(src_state)
    obj.load_state_dict(dst_state)

    return None

def cat_then_load_lora_experts(
    init_param: list,
    model: torch.nn.Module,
    ngpu: int,
    ignore_init_mismatch: bool=False,
):
    logging.info(f"Loading base model")
    # 0. load the librispeech100-finetuned model or original model
    src_state=torch.load(init_param[0], map_location=f"cuda:{torch.cuda.current_device()}" if ngpu > 0 else "cpu")
    model.load_state_dict(src_state, strict=ignore_init_mismatch)
    init_param.pop(0)

    logging.info(f"Loading cated expert")
    # 1. get src expert state
    expert_state_list=[]
    for p in init_param:
        src_state = torch.load(p, map_location=f"cuda:{torch.cuda.current_device()}" if ngpu > 0 else "cpu")
        src_state = {k.split("lora_A")[0]+"lora_A" if "lora_A" in k else k : v for k, v in src_state.items()}
        src_state = {k.split("lora_B")[0]+"lora_B" if "lora_B" in k else k : v for k, v in src_state.items()}
        expert_state_list.append(src_state)

    # 2. get dst model state
    dst_state=model.state_dict()
    dst_state={k:v for k,v in dst_state.items() if "kid" not in k}
    dst_state={k:v for k,v in dst_state.items() if "lora_A" in k or "lora_B" in k}

    # 3. cat each expert weight and cover the original weight
    for dst_key in dst_state.keys():
        expert_params = [expert_state[dst_key] for expert_state in expert_state_list]

        if 'lora_A' in dst_key:
            # lora_A shape: [r, in_features] -> cat at dim=0
            dst_state[dst_key] = torch.cat(expert_params, dim=0)
        elif 'lora_B' in dst_key:
            # lora_B shape: [out_features, r] -> cat at dim=1
            dst_state[dst_key] = torch.cat(expert_params, dim=1)

    # 4. load the cated weight
    model.load_state_dict(dst_state, strict=ignore_init_mismatch)
