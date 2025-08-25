def is_nemotron_hybrid(config):
    if hasattr(config, "hybrid_override_pattern"):
        return True
    return False


def is_mla(config):
    if hasattr(config, "kv_lora_rank"):
        assert hasattr(
            config, "qk_rope_head_dim"
        ), "both of kv_lora_rank and qk_rope_head_dim are required."
        return True
    return False


def is_nemotron_nas(config):
    if hasattr(config, "architectures") and config.architectures is not None:
        architectures = config.architectures
        return len(
            architectures) == 1 and architectures[0] == "DeciLMForCausalLM"
    return False
