import os


os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(int(os.cpu_count() or 1), 1)))
