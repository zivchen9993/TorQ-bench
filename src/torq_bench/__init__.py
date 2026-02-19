from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("torq-bench")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = ["PennyLaneQLayer", "PennyLaneComparison"]


def __getattr__(name):
    if name == "PennyLaneQLayer":
        from .pennylane_backend import PennyLaneQLayer
        globals()[name] = PennyLaneQLayer
        return PennyLaneQLayer
    if name == "PennyLaneComparison":
        from .PennyLaneComparison import PennyLaneComparison
        globals()[name] = PennyLaneComparison
        return PennyLaneComparison
    raise AttributeError(f"module {__name__} has no attribute {name}")
