from typing import List, Optional

import torch
from typeguard import typechecked

from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.layers.create_adapter_utils import (
    check_target_module_exists,
    get_submodules,
    replace_module,
)
from espnet2.layers.houlsby_adapter_layer import (
    Houlsby_Adapter,
    HoulsbyTransformerSentenceEncoderLayer,
    mark_only_houlsby_adapter_as_trainable,
)
import math
from typing import Union
from torch.nn.init import _calculate_correct_fan
try:
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2EncoderLayerStableLayerNorm,
    )

    is_transformers_available = True
except ImportError:
    is_transformers_available = False

try:
    import s3prl  # noqa
    from s3prl.upstream.wav2vec2.wav2vec2_model import TransformerSentenceEncoderLayer

    is_s3prl_available = True
except ImportError:
    is_s3prl_available = False

try:
    import loralib as lora

    is_lora_available = True
except ImportError:
    is_lora_available = False

# Generate random matrices of VeRA
# this code is copied from pefts.
def kaiming_init(
    tensor_or_shape,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Kaiming Uniform Initialisation adapted to accept a `torch.Generator` object for PRNG.

    Args:
        tensor_or_shape (`Union[torch.Tensor, tuple[int, ...]]`):
            Tensor to initialise, or shape of new tensor to create and then initialise.
        generator: (`torch.Generator`):
            Generator object that manages the state of the PRNG algorithm in use.

    Returns:
        `torch.Tensor`: The initialised tensor.
    """
    if isinstance(tensor_or_shape, tuple):
        tensor = torch.empty(tensor_or_shape)
    else:
        tensor = tensor_or_shape
    fan = _calculate_correct_fan(tensor, "fan_in")
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std

    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=generator)

@typechecked
def create_houlsby_adapter(
    model: torch.nn.Module,
    bottleneck: int = 32,
    target_layers: List[int] = [],
):
    if not is_transformers_available:
        raise ImportError(
            "`transformers` is not available. Please install it via `pip install"
            " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
            " && ./installers/install_transformers.sh`."
        )
    if not is_s3prl_available:
        raise ImportError(
            "Error: S3PRL is not properly installed."
            "Please install S3PRL: cd ${MAIN_ROOT}/tools && make s3prl.done"
        )
    assert hasattr(model, "frontend") and isinstance(
        model.frontend, S3prlFrontend
    ), "Only support S3PRL frontend now !!"

    is_traget_layer_exists = False
    key_list = [key for key, _ in model.named_modules()]
    num_layers = model.frontend.upstream.num_layers - 1
    if len(target_layers) == 0:
        target_layers = list(range(num_layers))

    for layer_idx in target_layers:

        key = f"frontend.upstream.upstream.model.encoder.layers.{layer_idx}"
        if key not in key_list:
            continue

        is_traget_layer_exists = True
        parent_module, target_name, target_module = get_submodules(model, key)
        new_module = create_new_houlsby_module(target_module, bottleneck)
        new_module.to(next(target_module.parameters()).device)
        setattr(parent_module, target_name, new_module)

    if not is_traget_layer_exists:
        raise ValueError(f"Target layers {target_layers} not found in the base model.")

    mark_only_houlsby_adapter_as_trainable(model, 'none')

@typechecked
def create_lora_adapter(
    model: torch.nn.Module,
    rank: int = 8,
    alpha: int = 8,
    dropout_rate: float = 0.0,
    target_modules: List[str] = ["query"],
    bias_type: Optional[str] = "none",
):
    """Create LoRA adapter for the base model.

    See: https://arxiv.org/pdf/2106.09685.pdf

    Args:
        model (torch.nn.Module): Base model to be adapted.
        rank (int): Rank of LoRA matrices. Defaults to 8.
        alpha (int): Constant number for LoRA scaling. Defaults to 8.
        dropout_rate (float): Dropout probability for LoRA layers. Defaults to 0.0.
        target_modules (List[str]): List of module(s) to apply LoRA adaptation.
            e.g. ["query", "key", "value"] for all layers,
            while ["encoder.encoders.blocks.0.attn.key"] for a specific layer.
        bias_type (str): Bias training type for LoRA adaptaion, can be
            one of ["none", "all", "lora_only"].
            "none" means not training any bias vectors;
            "all" means training all bias vectors, include LayerNorm biases;
            "lora_only" means only training bias vectors in LoRA adapted modules.


    """

    if not is_lora_available:
        raise ImportError(
            "Requiring loralib. Install loralib following: "
            "https://github.com/microsoft/LoRA"
        )

    is_traget_module_exists = False
    key_list = [key for key, _ in model.named_modules()]

    for key in key_list:
        if not check_target_module_exists(key, target_modules):
            continue

        # TODO(gituser) is this a good way to check the target module?
        # check_target_module_exists needs only one of the target modules
        # to be in the key, but what if one key exists and another doesn't?
        # Should this case raise an error?
        is_traget_module_exists = True

        parent_module, target_name, target_module = get_submodules(model, key)
        if not isinstance(target_module, lora.LoRALayer):
            new_module = create_new_lora_module(
                target_module, rank, alpha, dropout_rate
            )
            replace_module(parent_module, target_name, target_module, new_module)
        else:
            continue

    if not is_traget_module_exists:
        raise ValueError(
            f"Target modules {target_modules} not found in the base model."
        )
    
    lora.mark_only_lora_as_trainable(model, "lora", bias_type)

    # the developers claim that "This step can avoid merging LoRA weights again when loading pre-trained checkpoints"
    # for training stage, it works well
    # while for inference stage, it causes a bug that the kaiming-initialized LoRA weights are merged with the pretrained weights since model.eval() brings a merge operation
    # so model.eval() should be commented out.
    # model.eval()


@typechecked
def create_vera_adapter(
    model: torch.nn.Module,
    rank: int = 8,
    alpha: int = 8,
    dropout_rate: float = 0.0,
    target_modules: List[str] = ["query"],
    bias_type: Optional[str] = "none",
):
    """Create LoRA adapter for the base model.

    See: https://arxiv.org/pdf/2106.09685.pdf

    Args:
        model (torch.nn.Module): Base model to be adapted.
        rank (int): Rank of LoRA matrices. Defaults to 8.
        alpha (int): Constant number for LoRA scaling. Defaults to 8.
        dropout_rate (float): Dropout probability for LoRA layers. Defaults to 0.0.
        target_modules (List[str]): List of module(s) to apply LoRA adaptation.
            e.g. ["query", "key", "value"] for all layers,
            while ["encoder.encoders.blocks.0.attn.key"] for a specific layer.
        bias_type (str): Bias training type for LoRA adaptaion, can be
            one of ["none", "all", "lora_only"].
            "none" means not training any bias vectors;
            "all" means training all bias vectors, include LayerNorm biases;
            "lora_only" means only training bias vectors in LoRA adapted modules.


    """

    if not is_lora_available:
        raise ImportError(
            "Requiring loralib. Install loralib following: "
            "https://github.com/microsoft/LoRA"
        )

    is_traget_module_exists = False
    key_list = [key for key, _ in model.named_modules()]

    # get the max in_dim and out_dim, and then create vera_A and vera_B
    in_dims=[]
    out_dims=[]
    for key in key_list:
        if not check_target_module_exists(key, target_modules):
            continue
        target_module = model.get_submodule(key)
        in_dims.append(target_module.in_features)
        out_dims.append(target_module.out_features)

    generator = torch.Generator(device="cpu").manual_seed(0)
    vera_A = kaiming_init((rank, sorted(in_dims, reverse=True)[0]), generator=generator)
    vera_B = kaiming_init((sorted(out_dims, reverse=True)[0], rank), generator=generator)

    for key in key_list:
        if not check_target_module_exists(key, target_modules):
            continue

        # TODO(gituser) is this a good way to check the target module?
        # check_target_module_exists needs only one of the target modules
        # to be in the key, but what if one key exists and another doesn't?
        # Should this case raise an error?
        is_traget_module_exists = True

        parent_module, target_name, target_module = get_submodules(model, key)
        if not isinstance(target_module, lora.LoRALayer):
            new_module = create_new_vera_module(
                target_module, vera_A, vera_B, rank, alpha, dropout_rate
            )
            replace_module(parent_module, target_name, target_module, new_module)
        else:
            continue

    if not is_traget_module_exists:
        raise ValueError(
            f"Target modules {target_modules} not found in the base model."
        )
    
    lora.mark_only_lora_as_trainable(model, "vera", bias_type)

@typechecked
def create_dict_lora_adapter(
    model: torch.nn.Module,
    key_name: str,
    rank: int = 8,
    alpha: int = 8,
    dropout_rate: float = 0.0,
    target_modules: List[str] = ["query"],
    bias_type: Optional[str] = "none",
):
    """Create LoRA adapter for the base model.

    See: https://arxiv.org/pdf/2106.09685.pdf

    Args:
        model (torch.nn.Module): Base model to be adapted.
        rank (int): Rank of LoRA matrices. Defaults to 8.
        alpha (int): Constant number for LoRA scaling. Defaults to 8.
        dropout_rate (float): Dropout probability for LoRA layers. Defaults to 0.0.
        target_modules (List[str]): List of module(s) to apply LoRA adaptation.
            e.g. ["query", "key", "value"] for all layers,
            while ["encoder.encoders.blocks.0.attn.key"] for a specific layer.
        bias_type (str): Bias training type for LoRA adaptaion, can be
            one of ["none", "all", "lora_only"].
            "none" means not training any bias vectors;
            "all" means training all bias vectors, include LayerNorm biases;
            "lora_only" means only training bias vectors in LoRA adapted modules.

    """

    if not is_lora_available:
        raise ImportError(
            "Requiring loralib. Install loralib following: "
            "https://github.com/microsoft/LoRA"
        )

    is_traget_module_exists = False
    key_list = [key for key, _ in model.named_modules()]

    for key in key_list:
        if not check_target_module_exists(key, target_modules):
            continue

        # TODO(gituser) is this a good way to check the target module?
        # check_target_module_exists needs only one of the target modules
        # to be in the key, but what if one key exists and another doesn't?
        # Should this case raise an error?
        is_traget_module_exists = True

        parent_module, target_name, target_module = get_submodules(model, key)
        if not isinstance(target_module, lora.LoRALayer):
            new_module = create_new_dict_lora_module(
                target_module, rank, alpha, dropout_rate, key_name
            )
            replace_module(parent_module, target_name, target_module, new_module)
        else:
            continue

    if not is_traget_module_exists:
        raise ValueError(
            f"Target modules {target_modules} not found in the base model."
        )
    
    lora.mark_only_lora_as_trainable(model, "dictlora", bias_type)

@typechecked
def create_dict_lora4lanfusion_adapter(
    model: torch.nn.Module,
    key_name_list: List,
    rank: List,
    alpha: int = 8,
    dropout_rate: float = 0.0,
    target_modules: List[str] = ["query"],
    bias_type: Optional[str] = "none",
    domain=None, # make no sense, just prevent from complain
    Nbest=None, # make no sense, just prevent from complain
    expert_path=None, # make no sense, just prevent from complain
):
    """Create LoRA adapter for the base model.

    See: https://arxiv.org/pdf/2106.09685.pdf

    Args:
        model (torch.nn.Module): Base model to be adapted.
        rank (int): Rank of LoRA matrices. Defaults to 8.
        alpha (int): Constant number for LoRA scaling. Defaults to 8.
        dropout_rate (float): Dropout probability for LoRA layers. Defaults to 0.0.
        target_modules (List[str]): List of module(s) to apply LoRA adaptation.
            e.g. ["query", "key", "value"] for all layers,
            while ["encoder.encoders.blocks.0.attn.key"] for a specific layer.
        bias_type (str): Bias training type for LoRA adaptaion, can be
            one of ["none", "all", "lora_only"].
            "none" means not training any bias vectors;
            "all" means training all bias vectors, include LayerNorm biases;
            "lora_only" means only training bias vectors in LoRA adapted modules.

    """

    if not is_lora_available:
        raise ImportError(
            "Requiring loralib. Install loralib following: "
            "https://github.com/microsoft/LoRA"
        )

    is_traget_module_exists = False
    key_list = [key for key, _ in model.named_modules()]

    for key in key_list:
        if not check_target_module_exists(key, target_modules):
            continue

        # TODO(gituser) is this a good way to check the target module?
        # check_target_module_exists needs only one of the target modules
        # to be in the key, but what if one key exists and another doesn't?
        # Should this case raise an error?
        is_traget_module_exists = True

        parent_module, target_name, target_module = get_submodules(model, key)
        if not isinstance(target_module, lora.LoRALayer):
            new_module = create_new_dict_lora4lanfusion_module(
                target_module, rank, alpha, dropout_rate, key_name_list
            )
            replace_module(parent_module, target_name, target_module, new_module)
        else:
            continue

    if not is_traget_module_exists:
        raise ValueError(
            f"Target modules {target_modules} not found in the base model."
        )
    
    lora.mark_only_lora_as_trainable(model, "dictlora4lanfusion", bias_type)

@typechecked
def create_dict_lora4velora_adapter(
    model: torch.nn.Module,
    key_name_list: list,
    rank: List,
    alpha: int = 8,
    dropout_rate: float = 0.0,
    target_modules: List[str] = ["query"],
    bias_type: Optional[str] = "none",
    initial_type="ones",
    domain=None, # make no sense, just prevent from complain
    Nbest=None, # make no sense, just prevent from complain
    expert_path=None, # make no sense, just prevent from complain
):
    """Create LoRA adapter for the base model.

    See: https://arxiv.org/pdf/2106.09685.pdf

    Args:
        model (torch.nn.Module): Base model to be adapted.
        rank (int): Rank of LoRA matrices. Defaults to 8.
        alpha (int): Constant number for LoRA scaling. Defaults to 8.
        dropout_rate (float): Dropout probability for LoRA layers. Defaults to 0.0.
        target_modules (List[str]): List of module(s) to apply LoRA adaptation.
            e.g. ["query", "key", "value"] for all layers,
            while ["encoder.encoders.blocks.0.attn.key"] for a specific layer.
        bias_type (str): Bias training type for LoRA adaptaion, can be
            one of ["none", "all", "lora_only"].
            "none" means not training any bias vectors;
            "all" means training all bias vectors, include LayerNorm biases;
            "lora_only" means only training bias vectors in LoRA adapted modules.

    """

    if not is_lora_available:
        raise ImportError(
            "Requiring loralib. Install loralib following: "
            "https://github.com/microsoft/LoRA"
        )

    is_traget_module_exists = False
    key_list = [key for key, _ in model.named_modules()]

    for key in key_list:
        if not check_target_module_exists(key, target_modules):
            continue

        # TODO(gituser) is this a good way to check the target module?
        # check_target_module_exists needs only one of the target modules
        # to be in the key, but what if one key exists and another doesn't?
        # Should this case raise an error?
        is_traget_module_exists = True

        parent_module, target_name, target_module = get_submodules(model, key)
        if not isinstance(target_module, lora.LoRALayer):
            new_module = create_new_dict_lora4velora_module(
                target_module, rank, alpha, dropout_rate, key_name_list, initial_type
            )
            replace_module(parent_module, target_name, target_module, new_module)
        else:
            continue

    if not is_traget_module_exists:
        raise ValueError(
            f"Target modules {target_modules} not found in the base model."
        )
    
    lora.mark_only_lora_as_trainable(model, "dictlora4velora", bias_type)
    
@typechecked
def create_dict_lora4fastervelora_adapter(
    model: torch.nn.Module,
    key_name_list: list,
    rank: List,
    alpha: int = 8,
    dropout_rate: float = 0.0,
    target_modules: List[str] = ["query"],
    bias_type: Optional[str] = "none",
    initial_type="ones",
    domain=None, # make no sense, just prevent from complain
    Nbest=None, # make no sense, just prevent from complain
    expert_path=None, # make no sense, just prevent from complain
):
    """Create LoRA adapter for the base model.

    See: https://arxiv.org/pdf/2106.09685.pdf

    Args:
        model (torch.nn.Module): Base model to be adapted.
        rank (int): Rank of LoRA matrices. Defaults to 8.
        alpha (int): Constant number for LoRA scaling. Defaults to 8.
        dropout_rate (float): Dropout probability for LoRA layers. Defaults to 0.0.
        target_modules (List[str]): List of module(s) to apply LoRA adaptation.
            e.g. ["query", "key", "value"] for all layers,
            while ["encoder.encoders.blocks.0.attn.key"] for a specific layer.
        bias_type (str): Bias training type for LoRA adaptaion, can be
            one of ["none", "all", "lora_only"].
            "none" means not training any bias vectors;
            "all" means training all bias vectors, include LayerNorm biases;
            "lora_only" means only training bias vectors in LoRA adapted modules.

    """

    if not is_lora_available:
        raise ImportError(
            "Requiring loralib. Install loralib following: "
            "https://github.com/microsoft/LoRA"
        )

    is_traget_module_exists = False
    key_list = [key for key, _ in model.named_modules()]

    for key in key_list:
        if not check_target_module_exists(key, target_modules):
            continue

        # TODO(gituser) is this a good way to check the target module?
        # check_target_module_exists needs only one of the target modules
        # to be in the key, but what if one key exists and another doesn't?
        # Should this case raise an error?
        is_traget_module_exists = True

        parent_module, target_name, target_module = get_submodules(model, key)
        if not isinstance(target_module, lora.LoRALayer):
            new_module = create_new_dict_lora4fastervelora_module(
                target_module, rank, alpha, dropout_rate, key_name_list, initial_type
            )
            replace_module(parent_module, target_name, target_module, new_module)
        else:
            continue

    if not is_traget_module_exists:
        raise ValueError(
            f"Target modules {target_modules} not found in the base model."
        )
    
    lora.mark_only_lora_as_trainable(model, "dictlora4velora", bias_type)

@typechecked
def create_dict_lora4cat_adapter(
    model: torch.nn.Module,
    key_name_list: list,
    rank: List,
    alpha: int = 8,
    dropout_rate: float = 0.0,
    target_modules: List[str] = ["query"],
    bias_type: Optional[str] = "none",
    initial_type="ones",
    domain=None, # make no sense, just prevent from complain
    Nbest=None, # make no sense, just prevent from complain
    expert_path=None, # make no sense, just prevent from complain
):
    """Create LoRA adapter for the base model.

    See: https://arxiv.org/pdf/2106.09685.pdf

    Args:
        model (torch.nn.Module): Base model to be adapted.
        rank (int): Rank of LoRA matrices. Defaults to 8.
        alpha (int): Constant number for LoRA scaling. Defaults to 8.
        dropout_rate (float): Dropout probability for LoRA layers. Defaults to 0.0.
        target_modules (List[str]): List of module(s) to apply LoRA adaptation.
            e.g. ["query", "key", "value"] for all layers,
            while ["encoder.encoders.blocks.0.attn.key"] for a specific layer.
        bias_type (str): Bias training type for LoRA adaptaion, can be
            one of ["none", "all", "lora_only"].
            "none" means not training any bias vectors;
            "all" means training all bias vectors, include LayerNorm biases;
            "lora_only" means only training bias vectors in LoRA adapted modules.

    """

    if not is_lora_available:
        raise ImportError(
            "Requiring loralib. Install loralib following: "
            "https://github.com/microsoft/LoRA"
        )

    is_traget_module_exists = False
    key_list = [key for key, _ in model.named_modules()]

    for key in key_list:
        if not check_target_module_exists(key, target_modules):
            continue

        # TODO(gituser) is this a good way to check the target module?
        # check_target_module_exists needs only one of the target modules
        # to be in the key, but what if one key exists and another doesn't?
        # Should this case raise an error?
        is_traget_module_exists = True

        parent_module, target_name, target_module = get_submodules(model, key)
        if not isinstance(target_module, lora.LoRALayer):
            new_module = create_new_dict_lora4cat_module(
                target_module, rank, alpha, dropout_rate, key_name_list, initial_type
            )
            replace_module(parent_module, target_name, target_module, new_module)
        else:
            continue

    if not is_traget_module_exists:
        raise ValueError(
            f"Target modules {target_modules} not found in the base model."
        )
    
    lora.mark_only_lora_as_trainable(model, "dictlora4velora", bias_type)

@typechecked
def create_dict_lora4ecam_adapter(
    model: torch.nn.Module,
    rank: int,
    alpha: int = 8,
    dropout_rate: float = 0.0,
    target_modules: List[str] = ["query"],
    bias_type: Optional[str] = "none",
    domain=None,
    Nbest=None, # make no sense, just prevent from complain
    expert_path=None, # make no sense, just prevent from complain
):
    """Create LoRA adapter for the base model.

    See: https://arxiv.org/pdf/2106.09685.pdf

    Args:
        model (torch.nn.Module): Base model to be adapted.
        rank (int): Rank of LoRA matrices. Defaults to 8.
        alpha (int): Constant number for LoRA scaling. Defaults to 8.
        dropout_rate (float): Dropout probability for LoRA layers. Defaults to 0.0.
        target_modules (List[str]): List of module(s) to apply LoRA adaptation.
            e.g. ["query", "key", "value"] for all layers,
            while ["encoder.encoders.blocks.0.attn.key"] for a specific layer.
        bias_type (str): Bias training type for LoRA adaptaion, can be
            one of ["none", "all", "lora_only"].
            "none" means not training any bias vectors;
            "all" means training all bias vectors, include LayerNorm biases;
            "lora_only" means only training bias vectors in LoRA adapted modules.

    """

    if not is_lora_available:
        raise ImportError(
            "Requiring loralib. Install loralib following: "
            "https://github.com/microsoft/LoRA"
        )

    is_traget_module_exists = False
    key_list = [key for key, _ in model.named_modules()]

    for key in key_list:
        if not check_target_module_exists(key, target_modules):
            continue

        # TODO(gituser) is this a good way to check the target module?
        # check_target_module_exists needs only one of the target modules
        # to be in the key, but what if one key exists and another doesn't?
        # Should this case raise an error?
        is_traget_module_exists = True

        parent_module, target_name, target_module = get_submodules(model, key)
        if not isinstance(target_module, lora.LoRALayer):
            new_module = create_new_dict_lora4ecam_module(
                target_module, rank, alpha, dropout_rate, domain
            )
            replace_module(parent_module, target_name, target_module, new_module)
        else:
            continue

    if not is_traget_module_exists:
        raise ValueError(
            f"Target modules {target_modules} not found in the base model."
        )
    
    lora.mark_only_lora_as_trainable(model, "dictlora4ecam", bias_type)

@typechecked
def create_dict_lora4pcam_adapter(
    model: torch.nn.Module,
    key_name_list: List,
    rank: List,
    alpha: int = 8,
    dropout_rate: float = 0.0,
    target_modules: List[str] = ["query"],
    bias_type: Optional[str] = "none",
    domain=None,
    Nbest=None, # make no sense, just prevent from complain
    expert_path=None, # make no sense, just prevent from complain
):
    """Create LoRA adapter for the base model.

    See: https://arxiv.org/pdf/2106.09685.pdf

    Args:
        model (torch.nn.Module): Base model to be adapted.
        rank (int): Rank of LoRA matrices. Defaults to 8.
        alpha (int): Constant number for LoRA scaling. Defaults to 8.
        dropout_rate (float): Dropout probability for LoRA layers. Defaults to 0.0.
        target_modules (List[str]): List of module(s) to apply LoRA adaptation.
            e.g. ["query", "key", "value"] for all layers,
            while ["encoder.encoders.blocks.0.attn.key"] for a specific layer.
        bias_type (str): Bias training type for LoRA adaptaion, can be
            one of ["none", "all", "lora_only"].
            "none" means not training any bias vectors;
            "all" means training all bias vectors, include LayerNorm biases;
            "lora_only" means only training bias vectors in LoRA adapted modules.

    """

    if not is_lora_available:
        raise ImportError(
            "Requiring loralib. Install loralib following: "
            "https://github.com/microsoft/LoRA"
        )

    is_traget_module_exists = False
    key_list = [key for key, _ in model.named_modules()]

    for key in key_list:
        if not check_target_module_exists(key, target_modules):
            continue

        # TODO(gituser) is this a good way to check the target module?
        # check_target_module_exists needs only one of the target modules
        # to be in the key, but what if one key exists and another doesn't?
        # Should this case raise an error?
        is_traget_module_exists = True

        parent_module, target_name, target_module = get_submodules(model, key)
        if not isinstance(target_module, lora.LoRALayer):
            new_module = create_new_dict_lora4pcam_module(
                target_module, rank, alpha, dropout_rate, key_name_list, domain
            )
            replace_module(parent_module, target_name, target_module, new_module)
        else:
            continue

    if not is_traget_module_exists:
        raise ValueError(
            f"Target modules {target_modules} not found in the base model."
        )
    
    lora.mark_only_lora_as_trainable(model, "dictlora4pcam", bias_type)

@typechecked
def create_dict_lora4mole_adapter(
    model: torch.nn.Module,
    key_name_list: list,
    rank: List,
    alpha: int = 8,
    dropout_rate: float = 0.0,
    target_modules: List[str] = ["query"],
    bias_type: Optional[str] = "none",
    initial_type="ones",
    domain=None, # make no sense, just prevent from complain
    Nbest=None, # make no sense, just prevent from complain
    expert_path=None, # make no sense, just prevent from complain
):
    """Create LoRA adapter for the base model.

    See: https://arxiv.org/pdf/2106.09685.pdf

    Args:
        model (torch.nn.Module): Base model to be adapted.
        rank (int): Rank of LoRA matrices. Defaults to 8.
        alpha (int): Constant number for LoRA scaling. Defaults to 8.
        dropout_rate (float): Dropout probability for LoRA layers. Defaults to 0.0.
        target_modules (List[str]): List of module(s) to apply LoRA adaptation.
            e.g. ["query", "key", "value"] for all layers,
            while ["encoder.encoders.blocks.0.attn.key"] for a specific layer.
        bias_type (str): Bias training type for LoRA adaptaion, can be
            one of ["none", "all", "lora_only"].
            "none" means not training any bias vectors;
            "all" means training all bias vectors, include LayerNorm biases;
            "lora_only" means only training bias vectors in LoRA adapted modules.

    """

    if not is_lora_available:
        raise ImportError(
            "Requiring loralib. Install loralib following: "
            "https://github.com/microsoft/LoRA"
        )

    is_traget_module_exists = False
    key_list = [key for key, _ in model.named_modules()]

    for key in key_list:
        if not check_target_module_exists(key, target_modules):
            continue

        # TODO(gituser) is this a good way to check the target module?
        # check_target_module_exists needs only one of the target modules
        # to be in the key, but what if one key exists and another doesn't?
        # Should this case raise an error?
        is_traget_module_exists = True

        parent_module, target_name, target_module = get_submodules(model, key)
        if not isinstance(target_module, lora.LoRALayer):
            new_module = create_new_dict_lora4mole_module(
                target_module, rank, alpha, dropout_rate, key_name_list, initial_type
            )
            replace_module(parent_module, target_name, target_module, new_module)
        else:
            continue

    if not is_traget_module_exists:
        raise ValueError(
            f"Target modules {target_modules} not found in the base model."
        )
    
    lora.mark_only_lora_as_trainable(model, "dictlora4mole", bias_type)

@typechecked
def create_dict_lora4samd_adapter(
    model: torch.nn.Module,
    key_name_list: list,
    rank: List,
    alpha: int = 8,
    dropout_rate: float = 0.0,
    target_modules: List[str] = ["query"],
    bias_type: Optional[str] = "none",
    initial_type="ones",
    domain=None, # make no sense, just prevent from complain
    Nbest=None, # make no sense, just prevent from complain
    expert_path=None, # make no sense, just prevent from complain
):
    """Create LoRA adapter for the base model.

    See: https://arxiv.org/pdf/2106.09685.pdf

    Args:
        model (torch.nn.Module): Base model to be adapted.
        rank (int): Rank of LoRA matrices. Defaults to 8.
        alpha (int): Constant number for LoRA scaling. Defaults to 8.
        dropout_rate (float): Dropout probability for LoRA layers. Defaults to 0.0.
        target_modules (List[str]): List of module(s) to apply LoRA adaptation.
            e.g. ["query", "key", "value"] for all layers,
            while ["encoder.encoders.blocks.0.attn.key"] for a specific layer.
        bias_type (str): Bias training type for LoRA adaptaion, can be
            one of ["none", "all", "lora_only"].
            "none" means not training any bias vectors;
            "all" means training all bias vectors, include LayerNorm biases;
            "lora_only" means only training bias vectors in LoRA adapted modules.

    """

    if not is_lora_available:
        raise ImportError(
            "Requiring loralib. Install loralib following: "
            "https://github.com/microsoft/LoRA"
        )

    is_traget_module_exists = False
    key_list = [key for key, _ in model.named_modules()]

    for key in key_list:
        if not check_target_module_exists(key, target_modules):
            continue

        # TODO(gituser) is this a good way to check the target module?
        # check_target_module_exists needs only one of the target modules
        # to be in the key, but what if one key exists and another doesn't?
        # Should this case raise an error?
        is_traget_module_exists = True

        parent_module, target_name, target_module = get_submodules(model, key)
        if not isinstance(target_module, lora.LoRALayer):
            new_module = create_new_dict_lora4samd_module(
                target_module, rank, alpha, dropout_rate, key_name_list, initial_type
            )
            replace_module(parent_module, target_name, target_module, new_module)
        else:
            continue

    if not is_traget_module_exists:
        raise ValueError(
            f"Target modules {target_modules} not found in the base model."
        )
    
    lora.mark_only_lora_as_trainable(model, "dictlora", bias_type)

@typechecked
def create_moslora_adapter(
    model: torch.nn.Module,
    rank: int = 8,
    alpha: int = 8,
    dropout_rate: float = 0.0,
    target_modules: List[str] = ["query"],
    bias_type: Optional[str] = "none",
):
    """Create MosLoRA adapter for the base model.

    See: https://arxiv.org/pdf/2406.11909

    The args are same as LoRA

    Args:
        model (torch.nn.Module): Base model to be adapted.
        rank (int): Rank of LoRA matrices. Defaults to 8.
        alpha (int): Constant number for LoRA scaling. Defaults to 8.
        dropout_rate (float): Dropout probability for LoRA layers. Defaults to 0.0.
        target_modules (List[str]): List of module(s) to apply LoRA adaptation.
            e.g. ["query", "key", "value"] for all layers,
            while ["encoder.encoders.blocks.0.attn.key"] for a specific layer.
        bias_type (str): Bias training type for LoRA adaptaion, can be
            one of ["none", "all", "lora_only"].
            "none" means not training any bias vectors;
            "all" means training all bias vectors, include LayerNorm biases;
            "lora_only" means only training bias vectors in LoRA adapted modules.
    """

    if not is_lora_available:
        raise ImportError(
            "Requiring loralib. Install loralib following: "
            "https://github.com/microsoft/LoRA"
        )

    is_traget_module_exists = False
    key_list = [key for key, _ in model.named_modules()]

    for key in key_list:
        if not check_target_module_exists(key, target_modules):
            continue

        # TODO(gituser) is this a good way to check the target module?
        # check_target_module_exists needs only one of the target modules
        # to be in the key, but what if one key exists and another doesn't?
        # Should this case raise an error?
        is_traget_module_exists = True

        parent_module, target_name, target_module = get_submodules(model, key)
        if not isinstance(target_module, lora.LoRALayer):
            # this is the only difference between create_moslora_adapter and create_lora_adapter
            new_module = create_new_moslora_module(
                target_module, rank, alpha, dropout_rate
            )
            replace_module(parent_module, target_name, target_module, new_module)
        else:
            continue

    if not is_traget_module_exists:
        raise ValueError(
            f"Target modules {target_modules} not found in the base model."
        )
    
    lora.mark_only_lora_as_trainable(model, "lora", bias_type)

@typechecked
def create_melora_adapter(
    model: torch.nn.Module,
    rank: list = [2, 4, 6, 8],
    alpha: list = [2, 4, 6, 8],
    dropout_rate: float = 0.0,
    target_modules: List[str] = ["query"],
    bias_type: Optional[str] = "none",
):
    """Create MELoRA adapter for the base model.
    See: https://arxiv.org/pdf/2402.17263v2
    """

    if not is_lora_available:
        raise ImportError(
            "Requiring loralib. Install loralib following: "
            "https://github.com/microsoft/LoRA"
        )

    is_traget_module_exists = False
    key_list = [key for key, _ in model.named_modules()]

    for key in key_list:
        if not check_target_module_exists(key, target_modules):
            continue

        # TODO(gituser) is this a good way to check the target module?
        # check_target_module_exists needs only one of the target modules
        # to be in the key, but what if one key exists and another doesn't?
        # Should this case raise an error?
        is_traget_module_exists = True

        parent_module, target_name, target_module = get_submodules(model, key)
        if not isinstance(target_module, lora.LoRALayer):
            new_module = create_new_melora_module(
                target_module, rank, alpha, dropout_rate
            )
            replace_module(parent_module, target_name, target_module, new_module)
        else:
            continue

    if not is_traget_module_exists:
        raise ValueError(
            f"Target modules {target_modules} not found in the base model."
        )
    
    lora.mark_only_lora_as_trainable(model, bias_type)

@typechecked
def create_lora_houslby_adapter(
    model: torch.nn.Module,
    use_lora: bool = True,
    rank: int = 8,
    alpha: int = 8,
    dropout_rate: float = 0.0,
    # for adapter
    bottleneck: int = 128,
    adapterH_dropout: float = 0.0,
    target_modules: List[str] = ["query"], # for LoRA by default
    target_modules_for_adapterh: List[str] = ["attn.out", "mlp.2"], # for adapter by default
    bias_type: Optional[str] = "none",
):
    """Create LoRA adapter for the base model.

    See: https://arxiv.org/pdf/2106.09685.pdf

    Args:
        model (torch.nn.Module): Base model to be adapted.
        rank (int): Rank of LoRA matrices. Defaults to 8.
        alpha (int): Constant number for LoRA scaling. Defaults to 8.
        dropout_rate (float): Dropout probability for LoRA layers. Defaults to 0.0.
        target_modules (List[str]): List of module(s) to apply LoRA adaptation.
            e.g. ["query", "key", "value"] for all layers,
            while ["encoder.encoders.blocks.0.attn.key"] for a specific layer.
        bias_type (str): Bias training type for LoRA adaptaion, can be
            one of ["none", "all", "lora_only"].
            "none" means not training any bias vectors;
            "all" means training all bias vectors, include LayerNorm biases;
            "lora_only" means only training bias vectors in LoRA adapted modules.


    """

    if not is_lora_available:
        raise ImportError(
            "Requiring loralib. Install loralib following: "
            "https://github.com/microsoft/LoRA"
        )

    is_traget_module_exists = False
    key_list = [key for key, _ in model.named_modules()]

    for key in key_list:
        if not check_target_module_exists(key, target_modules):
            continue

        # TODO(gituser) is this a good way to check the target module?
        # check_target_module_exists needs only one of the target modules
        # to be in the key, but what if one key exists and another doesn't?
        # Should this case raise an error?
        is_traget_module_exists = True

        parent_module, target_name, target_module = get_submodules(model, key)

        if not isinstance(target_module, lora.LoRALayer):

            if any(element in key for element in target_modules_for_adapterh):
                use_houslby=True
                new_module = create_new_lora_houslby_module(
                    target_module, use_lora, rank, alpha, dropout_rate, use_houslby, bottleneck, adapterH_dropout
                )
            else:
                use_houslby=False
                new_module = create_new_lora_houslby_module(
                    target_module, use_lora, rank, alpha, dropout_rate, use_houslby, bottleneck, adapterH_dropout
                )

            replace_module(parent_module, target_name, target_module, new_module)
        else:
            continue

    if not is_traget_module_exists:
        raise ValueError(
            f"Target modules {target_modules} not found in the base model."
        )
    
    lora.mark_only_lora_as_trainable(model, bias_type)




@typechecked
def create_new_houlsby_module(target_module: torch.nn.Module, bottleneck: int):
    """Create a new houlsby adapter module for the given target module.

    Currently, only support:
    Wav2Vec2EncoderLayerStableLayerNorm &
    TransformerSentenceEncoderLayer
    """
    if isinstance(target_module, Wav2Vec2EncoderLayerStableLayerNorm):

        input_size = target_module.layer_norm.normalized_shape[0]
        target_module.bottleneck = bottleneck
        target_module.adapter_layer = Houlsby_Adapter(
            input_size=input_size, bottleneck=bottleneck
        )
        adapter_added_layer = target_module

    elif isinstance(target_module, TransformerSentenceEncoderLayer):

        if HoulsbyTransformerSentenceEncoderLayer is None:
            raise ImportError(
                "Error: S3PRL is not properly installed."
                "Please install S3PRL: cd ${MAIN_ROOT}/tools && make s3prl.done"
            )

        embedding_dim = target_module.embedding_dim
        ffn_embedding_dim = target_module.fc1.out_features
        num_attention_heads = target_module.self_attn.num_heads
        dropout = target_module.dropout1.p
        attention_dropout = target_module.self_attn.dropout_module.p
        activation_dropout = target_module.dropout2.p
        activation_fn = target_module.activation_fn.__name__
        layer_norm_first = target_module.layer_norm_first

        # initialize adapter-added transformer layer
        adapter_added_layer = HoulsbyTransformerSentenceEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            layer_norm_first=layer_norm_first,
            bottleneck=bottleneck,
        )

        # Get default requires_grad
        for n, p in adapter_added_layer.named_parameters():
            if "adapter" in n:
                continue
            p.requires_grad = eval(f"target_module.{n}").requires_grad

        # copy weights from the target module
        orig_state_dict = target_module.state_dict()
        adapter_added_layer.load_state_dict(orig_state_dict, strict=False)

        # Copy all hooks to the new layer
        for k, v in target_module.__dict__.items():
            if "hook" not in k:
                continue
            adapter_added_layer.__dict__[k] = v
    else:
        raise NotImplementedError(
            f"Target module {type(target_module)} is not supported."
        )
    return adapter_added_layer

@typechecked
def create_new_lora_module(
    target_module: torch.nn.Module, rank: int, alpha: int, dropout_rate: float
):
    """Create a new lora module for the given target module."""
    bias = hasattr(target_module, "bias") and target_module.bias is not None

    if isinstance(target_module, torch.nn.Linear):
        new_module = lora.Linear(
            target_module.in_features,
            target_module.out_features,
            bias=bias,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout_rate,
        )
    else:
        raise ValueError(
            f"Target module {target_module} is not supported. "
            f"Currently, only `torch.nn.Embedding`, `torch.nn.Conv2d` "
            f"`torch.nn.Linear` and are supported."
        )

    return new_module

@typechecked
def create_new_vera_module(
    target_module: torch.nn.Module, vera_A, vera_B, rank: int, alpha: int, dropout_rate: float
):
    """Create a new lora module for the given target module."""
    bias = hasattr(target_module, "bias") and target_module.bias is not None

    if isinstance(target_module, torch.nn.Linear):
        new_module = lora.LinearForVeRA(
            target_module.in_features,
            target_module.out_features,
            bias=bias,
            vera_A=vera_A,
            vera_B=vera_B,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout_rate,
        )
    else:
        raise ValueError(
            f"Target module {target_module} is not supported. "
            f"Currently, only `torch.nn.Embedding`, `torch.nn.Conv2d` "
            f"`torch.nn.Linear` and are supported."
        )

    return new_module

@typechecked
def create_new_dict_lora_module(
    target_module: torch.nn.Module, rank: int, alpha: int, dropout_rate: float, key_name: str,
):
    """Create a new lora module for the given target module."""
    bias = hasattr(target_module, "bias") and target_module.bias is not None

    if isinstance(target_module, torch.nn.Linear):
        new_module = lora.LinearForDictLoRA(
            target_module.in_features,
            target_module.out_features,
            bias=bias,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout_rate,
            key=key_name
        )
    else:
        raise ValueError(
            f"Target module {target_module} is not supported. "
            f"Currently, only `torch.nn.Embedding`, `torch.nn.Conv2d` "
            f"`torch.nn.Linear` and are supported."
        )

    return new_module

@typechecked
def create_new_dict_lora4lanfusion_module(
    target_module: torch.nn.Module, rank: List, alpha: int, dropout_rate: float, key_name_list: List,
):
    """Create a new lora module for the given target module."""
    bias = hasattr(target_module, "bias") and target_module.bias is not None

    if isinstance(target_module, torch.nn.Linear):
        new_module = lora.LinearForDictLoRA4LanFusion(
            target_module.in_features,
            target_module.out_features,
            bias=bias,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout_rate,
            key_list=key_name_list
        )
    else:
        raise ValueError(
            f"Target module {target_module} is not supported. "
            f"Currently, only `torch.nn.Embedding`, `torch.nn.Conv2d` "
            f"`torch.nn.Linear` and are supported."
        )

    return new_module

@typechecked
def create_new_dict_lora4velora_module(
    target_module: torch.nn.Module, rank: list, alpha: int, dropout_rate: float, key_name_list: List, initial_type
):
    """Create a new lora module for the given target module."""
    bias = hasattr(target_module, "bias") and target_module.bias is not None

    if isinstance(target_module, torch.nn.Linear):
        new_module = lora.LinearForDictLoRA4VeLoRA(
            target_module.in_features,
            target_module.out_features,
            bias=bias,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout_rate,
            key_list=key_name_list,
            initial_type=initial_type,
        )
    else:
        raise ValueError(
            f"Target module {target_module} is not supported. "
            f"Currently, only `torch.nn.Embedding`, `torch.nn.Conv2d` "
            f"`torch.nn.Linear` and are supported."
        )

    return new_module

@typechecked
def create_new_dict_lora4fastervelora_module(
    target_module: torch.nn.Module, rank: list, alpha: int, dropout_rate: float, key_name_list: List, initial_type
):
    """Create a new lora module for the given target module."""
    bias = hasattr(target_module, "bias") and target_module.bias is not None

    if isinstance(target_module, torch.nn.Linear):
        new_module = lora.LinearForDictLoRA4FasterVeLoRA(
            target_module.in_features,
            target_module.out_features,
            bias=bias,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout_rate,
            key_list=key_name_list,
            initial_type=initial_type,
        )
    else:
        raise ValueError(
            f"Target module {target_module} is not supported. "
            f"Currently, only `torch.nn.Embedding`, `torch.nn.Conv2d` "
            f"`torch.nn.Linear` and are supported."
        )

    return new_module

@typechecked
def create_new_dict_lora4cat_module(
    target_module: torch.nn.Module, rank: list, alpha: int, dropout_rate: float, key_name_list: List, initial_type
):
    """Create a new lora module for the given target module."""
    bias = hasattr(target_module, "bias") and target_module.bias is not None

    if isinstance(target_module, torch.nn.Linear):
        new_module = lora.LinearForDictLoRA4CAT(
            target_module.in_features,
            target_module.out_features,
            bias=bias,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout_rate,
            key_list=key_name_list,
            initial_type=initial_type,
        )
    else:
        raise ValueError(
            f"Target module {target_module} is not supported. "
            f"Currently, only `torch.nn.Embedding`, `torch.nn.Conv2d` "
            f"`torch.nn.Linear` and are supported."
        )

    return new_module

@typechecked
def create_new_dict_lora4ecam_module(
    target_module: torch.nn.Module, rank: int, alpha: int, dropout_rate: float, domain
):
    """Create a new lora module for the given target module."""
    bias = hasattr(target_module, "bias") and target_module.bias is not None

    if isinstance(target_module, torch.nn.Linear):
        new_module = lora.LinearForDictLoRA4ECAM(
            target_module.in_features,
            target_module.out_features,
            bias=bias,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout_rate,
            key=domain,
        )
    else:
        raise ValueError(
            f"Target module {target_module} is not supported. "
            f"Currently, only `torch.nn.Embedding`, `torch.nn.Conv2d` "
            f"`torch.nn.Linear` and are supported."
        )

    return new_module

@typechecked
def create_new_dict_lora4pcam_module(
    target_module: torch.nn.Module, rank: List, alpha: int, dropout_rate: float, key_name_list: List, domain
):
    """Create a new lora module for the given target module."""
    bias = hasattr(target_module, "bias") and target_module.bias is not None

    if isinstance(target_module, torch.nn.Linear):
        new_module = lora.LinearForDictLoRA4PCAM(
            target_module.in_features,
            target_module.out_features,
            bias=bias,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout_rate,
            key_list=key_name_list,
            domain=domain,
        )
    else:
        raise ValueError(
            f"Target module {target_module} is not supported. "
            f"Currently, only `torch.nn.Embedding`, `torch.nn.Conv2d` "
            f"`torch.nn.Linear` and are supported."
        )

    return new_module

@typechecked
def create_new_dict_lora4mole_module(
    target_module: torch.nn.Module, rank: list, alpha: int, dropout_rate: float, key_name_list: List, initial_type
):
    """Create a new lora module for the given target module."""
    bias = hasattr(target_module, "bias") and target_module.bias is not None

    if isinstance(target_module, torch.nn.Linear):
        new_module = lora.LinearForDictLoRA4MOLE(
            target_module.in_features,
            target_module.out_features,
            bias=bias,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout_rate,
            key_list=key_name_list,
            initial_type=initial_type,
        )
    else:
        raise ValueError(
            f"Target module {target_module} is not supported. "
            f"Currently, only `torch.nn.Embedding`, `torch.nn.Conv2d` "
            f"`torch.nn.Linear` and are supported."
        )

    return new_module

@typechecked
def create_new_dict_lora4samd_module(
    target_module: torch.nn.Module, rank: list, alpha: int, dropout_rate: float, key_name_list: List, initial_type
):
    """Create a new lora module for the given target module."""
    bias = hasattr(target_module, "bias") and target_module.bias is not None

    if isinstance(target_module, torch.nn.Linear):
        new_module = lora.LinearForDictLoRA4SAMD(
            target_module.in_features,
            target_module.out_features,
            bias=bias,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout_rate,
            key_list=key_name_list,
            initial_type=initial_type,
        )
    else:
        raise ValueError(
            f"Target module {target_module} is not supported. "
            f"Currently, only `torch.nn.Embedding`, `torch.nn.Conv2d` "
            f"`torch.nn.Linear` and are supported."
        )

    return new_module

@typechecked
def create_new_moslora_module(
    target_module: torch.nn.Module, rank: int, alpha: int, dropout_rate: float
):
    """Create a new lora module for the given target module."""
    bias = hasattr(target_module, "bias") and target_module.bias is not None

    if isinstance(target_module, torch.nn.Linear):
        new_module = lora.LinearForMosLoRA(
            target_module.in_features,
            target_module.out_features,
            bias=bias,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout_rate,
        )
    else:
        raise ValueError(
            f"Target module {target_module} is not supported. "
            f"Currently, only `torch.nn.Embedding`, `torch.nn.Conv2d` "
            f"`torch.nn.Linear` and are supported."
        )

    return new_module

@typechecked
def create_new_melora_module(
    target_module: torch.nn.Module, rank: list, alpha: list, dropout_rate: float
):
    """Create a new lora module for the given target module."""
    bias = hasattr(target_module, "bias") and target_module.bias is not None

    if isinstance(target_module, torch.nn.Linear):
        new_module = lora.LinearForMeLoRA(
            target_module.in_features,
            target_module.out_features,
            bias=bias,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout_rate,
        )
    else:
        raise ValueError(
            f"Target module {target_module} is not supported. "
            f"Currently, only `torch.nn.Embedding`, `torch.nn.Conv2d` "
            f"`torch.nn.Linear` and are supported."
        )

    return new_module

@typechecked
def create_new_lora_houslby_module(
    target_module: torch.nn.Module, use_lora: bool, rank: int, alpha: int, dropout_rate: float, use_houslby: bool, bottleneck: int, adapterH_dropout: float,
):
    """Create a new lora module for the given target module."""
    bias = hasattr(target_module, "bias") and target_module.bias is not None

    if isinstance(target_module, torch.nn.Linear):
        new_module = lora.LinearForLoRACombineAdapterH(
            target_module.in_features,
            target_module.out_features,
            bias=bias,
            use_lora=use_lora,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout_rate,
            use_houslby=use_houslby,
            bottleneck=bottleneck,
            adapterH_dropout=adapterH_dropout
        )
    else:
        raise ValueError(
            f"Target module {target_module} is not supported. "
            f"Currently, only `torch.nn.Embedding`, `torch.nn.Conv2d` "
            f"`torch.nn.Linear` and are supported."
        )

    return new_module
