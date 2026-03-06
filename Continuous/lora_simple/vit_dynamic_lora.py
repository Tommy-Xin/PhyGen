from typing import Dict, List, Tuple

from peft import LoraConfig, get_peft_model


VIT_TARGET_SUBMODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
    "mlp.fc1",
    "mlp.fc2",
)


def _linear_rank_schedule(
    layer_index: int,
    start_layer: int,
    end_layer: int,
    min_rank: int,
    max_rank: int,
) -> int:
    if end_layer <= start_layer:
        return int(max(1, max_rank))
    ratio = (layer_index - start_layer) / float(end_layer - start_layer)
    rank = min_rank + ratio * (max_rank - min_rank)
    return int(max(1, round(rank)))


def build_vit_dynamic_lora_patterns(
    model,
    last_n_layers: int,
    min_rank: int,
    max_rank: int,
    base_alpha: int,
) -> Tuple[List[str], Dict[str, int], Dict[str, int], Tuple[int, int]]:
    num_layers = len(model.vision_model.encoder.layers)
    if num_layers <= 0:
        raise ValueError("vision_model.encoder.layers is empty; cannot apply ViT LoRA.")

    use_last_n = min(max(1, int(last_n_layers)), num_layers)
    start_layer = num_layers - use_last_n
    end_layer = num_layers - 1

    target_modules: List[str] = []
    rank_pattern: Dict[str, int] = {}
    alpha_pattern: Dict[str, int] = {}

    for layer_idx in range(start_layer, num_layers):
        layer_rank = _linear_rank_schedule(
            layer_index=layer_idx,
            start_layer=start_layer,
            end_layer=end_layer,
            min_rank=min_rank,
            max_rank=max_rank,
        )

        # Keep alpha roughly proportional to rank to stabilize scaling.
        layer_alpha = max(1, int(round(base_alpha * (layer_rank / float(max(1, min_rank))))))

        for submodule in VIT_TARGET_SUBMODULES:
            module_name = f"vision_model.encoder.layers.{layer_idx}.{submodule}"
            target_modules.append(module_name)
            rank_pattern[module_name] = layer_rank
            alpha_pattern[module_name] = layer_alpha

    return target_modules, rank_pattern, alpha_pattern, (start_layer, end_layer)


def apply_dynamic_vit_lora(
    model,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_bias: str,
    last_n_layers: int,
    max_rank: int,
):
    min_rank = int(max(1, lora_r))
    max_rank = int(max(min_rank, max_rank))
    target_modules, rank_pattern, alpha_pattern, layer_range = build_vit_dynamic_lora_patterns(
        model=model,
        last_n_layers=last_n_layers,
        min_rank=min_rank,
        max_rank=max_rank,
        base_alpha=int(max(1, lora_alpha)),
    )

    lora_config = LoraConfig(
        r=min_rank,
        lora_alpha=int(max(1, lora_alpha)),
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
    )
    peft_model = get_peft_model(model, lora_config)
    return peft_model, layer_range, rank_pattern
