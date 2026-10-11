"""Prevent two CLI processes from awarding the same incomplete trial a second attempt."""
import fcntl
from contextlib import contextmanager


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/"execution.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("this run already has an active executor; do not start a parallel resume") from error
        yield
