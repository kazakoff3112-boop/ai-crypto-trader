"""Process-wide POSIX flock for shadow commands that may write decisions."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
from typing import Iterator


class ShadowLockBusy(RuntimeError):
    """Another shadow decision command owns this database's process lock."""


@contextmanager
def shadow_process_lock(database_path: Path) -> Iterator[None]:
    """Keep the same lock inode for the whole run; never unlink it on release."""
    lock_path = database_path.with_name(database_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ShadowLockBusy("Another shadow decision run is active") from None
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
