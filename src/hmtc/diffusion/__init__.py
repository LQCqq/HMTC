__all__ = ["HMTFSampler", "JointDiffusion"]


def __getattr__(name: str):
    # Avoid cycles between data contracts, model construction and schedules.
    if name == "JointDiffusion":
        from .joint import JointDiffusion
        return JointDiffusion
    if name == "HMTFSampler":
        from .sampler import HMTFSampler
        return HMTFSampler
    raise AttributeError(name)
