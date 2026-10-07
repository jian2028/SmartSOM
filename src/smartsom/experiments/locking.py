"""Persistent, nonblocking advisory locks on local Windows and POSIX files."""

import os


def acquire(stream):
    if os.name == "nt":
        import errno
        import msvcrt

        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise BlockingIOError(exc.errno, str(exc)) from exc
            raise
    else:
        import fcntl

        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def release(stream):
    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream, fcntl.LOCK_UN)
