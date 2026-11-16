"""Keeps research data off folders that sync to the cloud."""

import ctypes
import os
import socket
import sqlite3
from pathlib import Path

_LOOPBACK_HOSTS = ("localhost", "::1")


class CloudPathError(Exception):
    """A path for research data is inside a folder that syncs to the cloud."""


class NetworkBlockedError(OSError):
    """Something in this program tried to reach the internet."""


def _is_loopback(host) -> bool:
    if isinstance(host, bytes):
        host = host.decode(errors="replace")
    return isinstance(host, str) and (host in _LOOPBACK_HOSTS or host.startswith("127."))


def block_network() -> None:
    """Refuse connections made through Python's sockets. The models load from disk and need none.

    Compiled libraries and other programs aren't covered; scripts/firewall.ps1 is the
    Windows-level lock that also covers them.
    """
    if getattr(socket.socket.connect, "_blocks_network", False):
        return
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo

    def connect(self, address):
        if not _is_loopback(address[0] if isinstance(address, tuple) else None):
            raise NetworkBlockedError(f"Blocked a connection to {address}.")
        return original_connect(self, address)

    def connect_ex(self, address):
        if not _is_loopback(address[0] if isinstance(address, tuple) else None):
            raise NetworkBlockedError(f"Blocked a connection to {address}.")
        return original_connect_ex(self, address)

    def getaddrinfo(host, *args, **kwargs):
        if host is not None and not _is_loopback(host):
            raise NetworkBlockedError(f"Blocked a lookup of {host}.")
        return original_getaddrinfo(host, *args, **kwargs)

    connect._blocks_network = True
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = getaddrinfo


def _volume_label(path: Path) -> str:
    anchor = path.anchor
    if not anchor or os.name != "nt":
        return ""
    buffer = ctypes.create_unicode_buffer(261)
    ok = ctypes.windll.kernel32.GetVolumeInformationW(
        ctypes.c_wchar_p(anchor), buffer, len(buffer), None, None, None, None, 0
    )
    return buffer.value if ok else ""


def _google_drive_roots() -> list[Path]:
    """Local folders Google Drive for desktop is set up to sync."""
    local_appdata = os.environ.get("LOCALAPPDATA")
    if not local_appdata:
        return []
    database = Path(local_appdata) / "Google" / "DriveFS" / "root_preference_sqlite.db"
    if not database.exists():
        return []
    try:
        connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=2)
        try:
            rows = connection.execute("SELECT last_seen_absolute_path FROM roots").fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        # Can't tell what's synced, so assume the usual backup folder is.
        return [Path.home() / "Documents"]
    return [Path(row[0]) for row in rows if row[0]]


def _onedrive_roots() -> list[Path]:
    names = ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")
    return [Path(os.environ[name]) for name in names if os.environ.get(name)]


def cloud_reason(path: Path) -> str | None:
    """Return why a path syncs to the cloud, or None if it looks local."""
    resolved = Path(path).resolve()
    if "google" in _volume_label(resolved).lower():
        return f"{resolved.anchor} is a Google Drive volume"
    for root in _google_drive_roots():
        if resolved.is_relative_to(root.resolve()):
            return f"{root} is synced by Google Drive"
    for root in _onedrive_roots():
        if resolved.is_relative_to(root.resolve()):
            return f"{root} is synced by OneDrive"
    return None


def require_local(path: Path, purpose: str) -> None:
    reason = cloud_reason(path)
    if reason:
        raise CloudPathError(f"{reason}. Use a local folder for {purpose}.")
