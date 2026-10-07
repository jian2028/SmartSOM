"""Windows process identities and termination using verified kernel handles."""

import ctypes
from ctypes import wintypes

kernel = ctypes.WinDLL("kernel32", use_last_error=True)


class ProcessEntry(ctypes.Structure):
    _fields_ = [
        ("size", wintypes.DWORD),
        ("usage", wintypes.DWORD),
        ("pid", wintypes.DWORD),
        ("heap", ctypes.c_size_t),
        ("module", wintypes.DWORD),
        ("threads", wintypes.DWORD),
        ("parent", wintypes.DWORD),
        ("priority", wintypes.LONG),
        ("flags", wintypes.DWORD),
        ("exe", wintypes.WCHAR * 260),
    ]


def _function(name, arguments, result):
    function = getattr(kernel, name)
    function.argtypes = arguments
    function.restype = result
    return function


_open = _function(
    "OpenProcess", [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE
)
_close = _function("CloseHandle", [wintypes.HANDLE], wintypes.BOOL)
_snapshot = _function(
    "CreateToolhelp32Snapshot", [wintypes.DWORD, wintypes.DWORD], wintypes.HANDLE
)
_first = _function(
    "Process32FirstW", [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)], wintypes.BOOL
)
_next = _function(
    "Process32NextW", [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)], wintypes.BOOL
)
_times = _function(
    "GetProcessTimes",
    [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4,
    wintypes.BOOL,
)
_wait = _function(
    "WaitForSingleObject", [wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD
)
_terminate = _function(
    "TerminateProcess", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL
)


def _created(handle):
    values = [wintypes.FILETIME() for _ in range(4)]
    if not _times(handle, *(ctypes.byref(value) for value in values)):
        raise ctypes.WinError(ctypes.get_last_error())
    return str((values[0].dwHighDateTime << 32) | values[0].dwLowDateTime)


def birth(pid):
    handle = _open(0x1000 | 0x00100000, False, pid)
    if not handle:
        return None
    try:
        if _wait(handle, 0) != 258:  # WAIT_TIMEOUT means still running.
            return None
        return _created(handle)
    finally:
        _close(handle)


def processes():
    snapshot = _snapshot(2, 0)  # TH32CS_SNAPPROCESS
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    rows = {}
    try:
        entry = ProcessEntry()
        entry.size = ctypes.sizeof(entry)
        available = _first(snapshot, ctypes.byref(entry))
        while available:
            created = birth(entry.pid)
            if created is not None:
                rows[entry.pid] = {
                    "pid": entry.pid,
                    "parent": entry.parent,
                    "created": created,
                    "started_at": created,
                    "state": "R",
                }
            available = _next(snapshot, ctypes.byref(entry))
        if ctypes.get_last_error() != 18:  # ERROR_NO_MORE_FILES
            raise ctypes.WinError(ctypes.get_last_error())
        return rows
    finally:
        _close(snapshot)


def terminate(identity):
    # Verify and act through the SAME handle: a reused PID cannot be killed.
    handle = _open(0x1000 | 0x00100000 | 1, False, identity["pid"])
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # Process has exited.
            return
        raise ctypes.WinError(error)
    try:
        if _wait(handle, 0) != 258 or _created(handle) != identity["created"]:
            return
        if not _terminate(handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        _close(handle)
