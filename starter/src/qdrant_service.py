"""Start the pinned official Qdrant Windows binary locally when it is not running."""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[2]
QDRANT_VERSION = "1.20.0"
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
TOOL_DIR = ROOT / "starter" / "tools" / "qdrant"
QDRANT_EXE = TOOL_DIR / "qdrant.exe"
QDRANT_ARCHIVE_URL = (
    f"https://github.com/qdrant/qdrant/releases/download/v{QDRANT_VERSION}/"
    "qdrant-x86_64-pc-windows-msvc.zip"
)


def _download_qdrant() -> None:
    TOOL_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="qdrant-", suffix=".zip", dir=TOOL_DIR, delete=False) as handle:
        archive_path = Path(handle.name)
    try:
        request = urllib.request.Request(QDRANT_ARCHIVE_URL, headers={"User-Agent": "day06-local-lab/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response, archive_path.open("wb") as target:
            target.write(response.read())
        with zipfile.ZipFile(archive_path) as archive:
            root = TOOL_DIR.resolve()
            for member in archive.infolist():
                destination = (TOOL_DIR / member.filename).resolve()
                if not destination.is_relative_to(root):
                    raise RuntimeError(f"Unexpected path in official Qdrant archive: {member.filename}")
            archive.extractall(TOOL_DIR)
        if not QDRANT_EXE.exists():
            nested = list(TOOL_DIR.rglob("qdrant.exe"))
            if nested:
                nested[0].replace(QDRANT_EXE)
        if not QDRANT_EXE.exists():
            raise RuntimeError("The Qdrant release archive did not contain qdrant.exe")
    finally:
        archive_path.unlink(missing_ok=True)


def _is_healthy() -> bool:
    try:
        response = httpx.get(f"{QDRANT_URL}/healthz", timeout=1.5)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def ensure_qdrant(storage_path: Path, startup_timeout: float = 60.0) -> None:
    """Ensure localhost Qdrant is serving; never silently substitute another backend."""
    if _is_healthy():
        return
    if os.name != "nt":
        raise RuntimeError("Qdrant is not reachable. Start Qdrant locally or configure QDRANT_URL.")
    if not QDRANT_EXE.exists():
        _download_qdrant()

    storage_path = storage_path.resolve()
    storage_path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "QDRANT__SERVICE__HOST": "127.0.0.1",
        "QDRANT__SERVICE__HTTP_PORT": "6333",
        "QDRANT__STORAGE__STORAGE_PATH": str(storage_path),
        "QDRANT__TELEMETRY_DISABLED": "true",
    })
    log_path = ROOT / "results" / "qdrant_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            [str(QDRANT_EXE)],
            cwd=str(storage_path),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    (ROOT / "results" / "qdrant.pid").write_text(str(process.pid), encoding="ascii")

    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if _is_healthy():
            return
        if process.poll() is not None:
            details = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
            raise RuntimeError(f"Qdrant exited with code {process.returncode}. Log tail:\n{details}")
        time.sleep(0.5)
    raise TimeoutError(f"Qdrant did not become healthy at {QDRANT_URL}; inspect {log_path}")
