"""
Scheduler factory — maps configuration enum values to diffusers schedulers.
"""

from diffusers import (
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)


SCHEDULER_MAP = {
    "euler_a": EulerAncestralDiscreteScheduler,
    "euler": EulerDiscreteScheduler,
    "dpm++_2m": DPMSolverMultistepScheduler,
    "dpm++_2m_karras": lambda config: DPMSolverMultistepScheduler.from_config(
        config, use_karras_sigmas=True
    ),
    "ddim": DDIMScheduler,
    "lms": LMSDiscreteScheduler,
    "pndm": PNDMScheduler,
}


def build_scheduler(scheduler_name: str, scheduler_config):
    """
    Build a scheduler instance from a name string and the pipeline's
    existing scheduler config.
    """
    entry = SCHEDULER_MAP.get(scheduler_name)
    if entry is None:
        raise ValueError(
            f"Unknown scheduler '{scheduler_name}'. "
            f"Available: {list(SCHEDULER_MAP.keys())}"
        )

    if callable(entry) and not isinstance(entry, type):
        # Lambda factory (e.g. Karras variant)
        return entry(scheduler_config)
    else:
        return entry.from_config(scheduler_config)
