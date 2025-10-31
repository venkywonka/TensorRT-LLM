"""TensorRT-LLM Recipe System for Optimized Inference Configurations.

This module provides a recipe-based configuration system for TensorRT-LLM,
allowing users to generate optimized configurations for specific inference
scenarios.
"""

from .profiles import PROFILE_REGISTRY, ProfileBase, get_profile, register_profile
from .matcher import detect_profile, match_recipe, compute_from_scenario
from .validator import validate_scenario, validate_config

__all__ = [
    'PROFILE_REGISTRY',
    'ProfileBase',
    'get_profile',
    'register_profile',
    'detect_profile',
    'match_recipe',
    'compute_from_scenario',
    'validate_scenario',
    'validate_config',
]


