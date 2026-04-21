import os
from contextvars import ContextVar
from pydantic import BaseModel
from typing import Optional

# Base Configuration
GLOBAL_MODEL_NAME = "qwen/qwen3.5-122b-a10b"

DEFAULT_CONFIG = {
    "classifier_model_name": GLOBAL_MODEL_NAME,
    "fact_checker_model_name": GLOBAL_MODEL_NAME,
    "classifier_n_samples": 5,
    "fact_checker_n_samples": 3,
    "fact_checker_max_loops": 3,
    "max_completion_tokens": 4096,
    "reasoning_effort": "low",
    "verbose_logging": os.getenv("VERBOSE_LOGGING", "false").lower() == "true",
}


# Request-scoped Configuration
class PipelineConfig(BaseModel):
    classifier_model_name: Optional[str] = None
    fact_checker_model_name: Optional[str] = None

    classifier_n_samples: Optional[int] = None
    fact_checker_n_samples: Optional[int] = None
    fact_checker_max_loops: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    classifier_reasoning_effort: Optional[str] = None
    fact_checker_reasoning_effort: Optional[str] = None
    verbose_logging: Optional[bool] = None


# state for the current async context only
_current_configvar: ContextVar[Optional[PipelineConfig]] = ContextVar(
    "pipeline_config", default=None
)


def set_config(config_override: PipelineConfig):
    """Set the configuration for the current request context."""
    return _current_configvar.set(config_override)


def get_config_val(key: str):
    """
    Get a configuration value.
    Prioritizes request-scoped overrides, falls back to DEFAULT_CONFIG.
    """
    current = _current_configvar.get()
    if current is not None:
        val = getattr(current, key, None)
        if val is not None:
            return val
    return DEFAULT_CONFIG.get(key)


# Helpers to access properties
@property
def CLASSIFIER_MODEL_NAME():
    return get_config_val("classifier_model_name")


@property
def FACT_CHECKER_MODEL_NAME():
    return get_config_val("fact_checker_model_name")





@property
def CLASSIFIER_N_SAMPLES():
    return get_config_val("classifier_n_samples")


@property
def FACT_CHECKER_N_SAMPLES():
    return get_config_val("fact_checker_n_samples")


@property
def FACT_CHECKER_MAX_LOOPS():
    return get_config_val("fact_checker_max_loops")


@property
def MAX_COMPLETION_TOKENS():
    return get_config_val("max_completion_tokens")


def get_llm_kwargs(component: str = None) -> dict:
    """Returns extra kwargs for LLM API calls based on config."""
    kwargs = {}

    effort = None
    if component:
        effort = get_config_val(f"{component}_reasoning_effort")
    if not effort:
        effort = get_config_val("reasoning_effort")

    if effort:
        kwargs["reasoning_effort"] = effort

    return kwargs
