import os
import warnings


def configure_runtime_environment() -> None:
    cpu_count = os.cpu_count() or 1
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(int(cpu_count), 1)))
    warnings.filterwarnings(
        "ignore",
        message=".*Could not find the number of physical cores.*",
        category=UserWarning,
    )
    try:
        from joblib.externals.loky.backend import context as loky_context

        cpu_count_int = max(int(cpu_count), 1)

        def _safe_count_physical_cores():
            return cpu_count_int, None

        loky_context._count_physical_cores = _safe_count_physical_cores
    except Exception:
        pass
