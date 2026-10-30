"""Bounded, order-preserving parallel maps for imports (threads) and raster blocks (spawned processes).

Every helper yields results in the order of the input items, whatever order the workers finish in, and keeps
at most ``max_pending`` results in flight, so the caller (always the main process) writes outputs
sequentially and deterministically and memory stays bounded by ``workers x block``. With ``workers <= 1`` no
pool is created and the items are processed in the calling thread, i.e. the single-process behaviour.

Worker processes are started with the ``spawn`` method (safe with GDAL, OpenMP and threads, identical on Linux
and macOS); worker functions and initialisers must therefore be importable module-level callables.
"""
from __future__ import annotations

import os
import random
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

#: Politeness cap for concurrent HTTP readers (Earth Search / Planetary Computer): never more than this.
MAX_IMPORT_WORKERS = 8
#: Default number of concurrent scene imports (`resources.import_workers`); 1 reproduces the sequential import.
DEFAULT_IMPORT_WORKERS = 4
#: Default cap of automatic worker counts; memory per worker (model copy, block buffers) grows with workers.
AUTO_WORKER_CAP = 4

CGROUP_CPU_FILES = ("/sys/fs/cgroup/cpu.max",                                         # cgroup v2: "quota period"
                    ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"))  # v1


def available_cpus(cgroup_files=CGROUP_CPU_FILES) -> tuple[float, str]:
    """CPUs this process may use: the CPU affinity set, or a smaller cgroup quota (``docker run --cpus``).

    Returns ``(cpus, source)``; ``source`` is ``"cpu_count"``, ``"affinity"`` or the cgroup file that lowered it.
    """
    try:
        cpus, source = float(len(os.sched_getaffinity(0))), "affinity"
    except (AttributeError, OSError):
        cpus, source = float(os.cpu_count() or 1), "cpu_count"
    for entry in cgroup_files:
        try:
            if isinstance(entry, str):
                quota, period = Path(entry).read_text(encoding="ascii").split()[:2]
                name = entry
            else:
                quota, period = (Path(f).read_text(encoding="ascii").strip() for f in entry)
                name = entry[0]
        except (OSError, ValueError):
            continue
        try:
            limit = int(quota) / int(period)
        except ValueError:                      # "max": no quota
            break
        if 0 < limit < cpus:
            cpus, source = limit, name
        break
    return cpus, source


def auto_workers(cap: int = AUTO_WORKER_CAP, cpus: float | None = None) -> int:
    """``min(usable CPUs, cap)``, at least 1."""
    cpus = available_cpus()[0] if cpus is None else cpus
    return max(1, min(int(cap), int(cpus)))


def resolve_workers(workers: int | None = None, n_jobs: int | None = None, cap: int = AUTO_WORKER_CAP,
                    cpus: float | None = None) -> int:
    """Worker count: explicit `workers` (>= 1), else a positive `n_jobs`, else ``min(usable CPUs, cap)``.

    An explicit value is used as given (it may exceed the CPU count); `n_jobs` <= 0 (e.g. scikit-learn's -1
    for "all cores") means "not set" here because block workers hold a model copy each, so the automatic cap
    applies.
    """
    if workers is not None:
        return max(1, int(workers))
    if n_jobs is not None and int(n_jobs) > 0:
        return int(n_jobs)
    return auto_workers(cap, cpus)


def import_workers(workers: int | None = None) -> int:
    """Concurrent scene imports: `workers` (default `DEFAULT_IMPORT_WORKERS`), clamped to 1..`MAX_IMPORT_WORKERS`."""
    return max(1, min(MAX_IMPORT_WORKERS, int(DEFAULT_IMPORT_WORKERS if workers is None else workers)))


def ordered_map(fn: Callable, items: Iterable, workers: int = 1, kind: str = "thread",
                max_pending: int | None = None, initializer: Callable | None = None, initargs: tuple = ()) -> Iterator:
    """Yield ``fn(item)`` for every item, in input order, computed by up to `workers` threads or processes.

    At most `max_pending` (default ``2 x workers``) results are submitted or buffered at any time. An exception
    raised by `fn` is re-raised at the position of its item (earlier results were already yielded) and the
    pending work is cancelled. With ``workers <= 1`` this is a plain in-thread loop (`initializer` is not
    called: callers keep their single-process path in that case).
    """
    if kind not in ("thread", "process"):
        raise ValueError("kind must be 'thread' or 'process'")
    workers = int(workers)
    if workers <= 1:
        for item in items:
            yield fn(item)
        return
    max_pending = max(1, int(max_pending or 2 * workers))
    if kind == "thread":
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="geocrop", initializer=initializer,
                                  initargs=initargs)
    else:
        import multiprocessing
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=initializer, initargs=initargs)
    pending: deque[Future] = deque()
    try:
        for item in items:
            pending.append(pool.submit(fn, item))
            if len(pending) >= max_pending:
                yield pending.popleft().result()
        while pending:
            value = pending.popleft().result()
            if not pending:
                pool.shutdown(wait=True)       # all work is done: release the workers before the last yield, so a
            yield value                        # caller that stops iterating here does not leave a pool behind
    finally:
        for f in pending:
            f.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def limit_threads(n: int = 1) -> None:
    """Limit BLAS/OpenMP threads of this process (worker processes: one thread each, no oversubscription)."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(n)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:      # scikit-learn depends on it; without it the environment variables apply to new libraries
        return
    threadpool_limits(limits=n)


TRANSIENT_MARKERS = ("429", "503", "too many requests", "service unavailable", "slowdown", "slow down",
                     "temporarily unavailable", "rate limit")


def is_transient(exc: BaseException) -> bool:
    """A throttling / overload answer of a server (HTTP 429, 503) worth retrying after a pause."""
    text = str(exc).lower()
    return any(m in text for m in TRANSIENT_MARKERS)


def with_backoff(fn: Callable, attempts: int = 4, base_delay: float = 2.0, max_delay: float = 60.0,
                 sleep: Callable[[float], None] | None = None, retry_on: type[BaseException] | tuple = Exception):
    """Wrap `fn` so that throttling errors (`is_transient`) are retried with exponential backoff and jitter.

    Delays are ``base_delay x 2^k`` (capped at `max_delay`) plus up to 25 % jitter, so concurrent workers that
    were throttled together do not retry together. Other errors, and the last transient one, propagate.
    """
    def wrapped(*args, **kwargs):
        for k in range(attempts):
            try:
                return fn(*args, **kwargs)
            except retry_on as exc:
                if k == attempts - 1 or not is_transient(exc):
                    raise
                delay = min(max_delay, base_delay * 2 ** k)
                (sleep or time.sleep)(delay + random.uniform(0, 0.25 * delay))
    return wrapped
