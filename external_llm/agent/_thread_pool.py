"""Global ThreadPoolExecutor — eliminates pool create/destroy overhead (5-15ms/call)."""
import atexit
import os
from concurrent.futures import ThreadPoolExecutor

# Worker count scales with the host so high-core machines get more parallel
# tool throughput. Tool work is overwhelmingly I/O-bound (LLM calls, bash,
# browser), so we mirror CPython's ThreadPoolExecutor default rather than a
# hard-coded small constant. Floor of 4 keeps low-core containers usable and
# preserves the original minimum behaviour.
_default_workers = max(4, min(32, (os.cpu_count() or 1) + 4))
shared_pool = ThreadPoolExecutor(max_workers=_default_workers)
atexit.register(shared_pool.shutdown, wait=False)
