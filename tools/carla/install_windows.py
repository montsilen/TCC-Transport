"""Download the official CARLA 0.9.15 Windows package, resume, and extract.

Uses only Python's standard library. No changes to system proxy settings.
"""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
URL = "https://carla-releases.s3.us-east-005.backblazeb2.com/Windows/CARLA_0.9.15.zip"


def opener(direct):
    return urllib.request.build_opener(urllib.request.ProxyHandler({})) if direct else urllib.request.build_opener()


def pypi_metadata(name, version, direct):
    for attempt in range(8):
        try:
            with opener(direct).open("https://pypi.org/pypi/{}/{}/json".format(name, version), timeout=20) as r:
                return json.load(r)
        except (OSError, http.client.HTTPException, ValueError):
            if attempt == 7:
                raise
            time.sleep(1)


def download(args):
    archive = args.root / "downloads" / "CARLA_0.9.15.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with opener(args.direct).open(urllib.request.Request(URL, method="HEAD"), timeout=30) as r:
        size = int(r.headers["Content-Length"])
    chunk = 16 * 1024 * 1024
    parts = (size + chunk - 1) // chunk
    if archive.exists() and archive.stat().st_size == size:
        print("Archive already downloaded:", archive, flush=True)
        return archive
    # Parts plus assembled archive coexist until extraction; leave room for unpacking.
    if shutil.disk_usage(args.root).free < size * 4:
        raise RuntimeError("Need at least {:.1f} GB free for download and extraction".format(size * 4 / 1e9))

    def fetch(i):
        part = archive.with_suffix(".part{:04d}".format(i))
        start, end = i * chunk, min((i + 1) * chunk, size) - 1
        expected = end - start + 1
        for attempt in range(30):
            offset = part.stat().st_size if part.exists() else 0
            if offset == expected:
                return
            if offset > expected:
                raise RuntimeError("Oversized partial file: {}".format(part))
            req = urllib.request.Request(URL, headers={"Range": "bytes={}-{}".format(start + offset, end)})
            try:
                with opener(args.direct).open(req, timeout=30) as r:
                    required_range = "bytes {}-{}/{}".format(start + offset, end, size)
                    if r.status != 206 or r.headers.get("Content-Range") != required_range:
                        raise RuntimeError("Unexpected range response: {}".format(r.headers.get("Content-Range")))
                    with part.open("ab") as f:
                        while True:
                            block = r.read(64 * 1024)
                            if not block:
                                break
                            f.write(block)
                if part.stat().st_size == expected:
                    return
            except (OSError, RuntimeError, http.client.HTTPException) as exc:
                if attempt == 29:
                    raise RuntimeError("Part {} incomplete; rerun to resume: {}".format(i, exc)) from exc
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("Incomplete part {}".format(i))

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, i) for i in range(parts)]
        pending = set(futures)
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=15, return_when=concurrent.futures.FIRST_EXCEPTION)
            for f in done:
                f.result()
            received = sum(p.stat().st_size for p in archive.parent.glob("CARLA_0.9.15.part*"))
            print("Download {:.1f}% ({:.2f}/{:.2f} GB), elapsed {:.0f}s".format(100 * received / size, received / 1e9, size / 1e9, time.monotonic() - started), flush=True)
    digest = hashlib.sha256()
    with archive.open("wb") as target:
        for i in range(parts):
            with archive.with_suffix(".part{:04d}".format(i)).open("rb") as source:
                while True:
                    block = source.read(4 * 1024 * 1024)
                    if not block:
                        break
                    target.write(block)
                    digest.update(block)
    archive.with_suffix(".sha256").write_text(digest.hexdigest() + "  " + archive.name + "\n")
    # Checksum is a local integrity record, not an upstream authenticity signature.
    return archive


def extract(archive, root):
    destination = (root / "WindowsNoEditor").resolve()
    with zipfile.ZipFile(archive) as z:
        unpacked = sum(member.file_size for member in z.infolist())
        if shutil.disk_usage(root).free < unpacked:
            raise RuntimeError("Need {:.1f} GB free to extract the CARLA archive".format(unpacked / 1e9))
        bad = z.testzip()
        if bad:
            raise RuntimeError("ZIP CRC check failed: " + bad)
        for member in z.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError("Unsafe archive path: " + member.filename)
        # The official package can contain a WindowsNoEditor top-level folder.
        names = {member.filename.replace("\\", "/") for member in z.infolist()}
        prefixed = "WindowsNoEditor/CarlaUE4.exe" in names
        if prefixed and "CarlaUE4.exe" in names:
            raise RuntimeError("Archive contains ambiguous executable locations")
        z.extractall(root if prefixed else destination)
    if not (destination / "CarlaUE4.exe").is_file():
        raise RuntimeError("Extraction finished but CarlaUE4.exe was not found")
    print("CARLA_ROOT=" + str(destination), flush=True)


def wheels(args):
    target = args.root / "wheels"
    target.mkdir(parents=True, exist_ok=True)
    for name, version in [("carla", "0.9.15"), ("numpy", "1.24.4"), ("pygame", "2.5.2")]:
        metadata = pypi_metadata(name, version, args.direct)
        item = next(a for a in metadata["urls"] if "cp38" in a["filename"] and "win_amd64" in a["filename"])
        wheel_url = item["url"]
        if args.wheel_mirror:
            wheel_url = wheel_url.replace("https://files.pythonhosted.org/", args.wheel_mirror.rstrip("/") + "/")
        path = target / item["filename"]
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != item["digests"]["sha256"]:
            partial = path.with_suffix(".download")
            for attempt in range(20):
                offset = partial.stat().st_size if partial.exists() else 0
                if offset == item["size"]:
                    break
                if offset > item["size"]:
                    raise RuntimeError("Oversized partial wheel: " + str(partial))
                headers = {"Range": "bytes={}-".format(offset)} if offset else {}
                try:
                    with opener(args.direct).open(urllib.request.Request(wheel_url, headers=headers), timeout=20) as r:
                        if offset and r.status != 206:
                            raise RuntimeError("Wheel host did not honor resume request")
                        with partial.open("ab") as f:
                            while True:
                                data = r.read(64 * 1024)
                                if not data:
                                    break
                                f.write(data)
                    if partial.stat().st_size == item["size"]:
                        break
                except (OSError, http.client.HTTPException) as exc:
                    print("Wheel download retry {}: {}".format(attempt + 1, exc), flush=True)
            if not partial.exists() or hashlib.sha256(partial.read_bytes()).hexdigest() != item["digests"]["sha256"]:
                raise RuntimeError("Wheel checksum mismatch: " + item["filename"])
            partial.replace(path)
        print("Verified wheel:", path, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT / "simulation" / "carla")
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--direct", action="store_true", help="Ignore proxy environment for this downloader only")
    p.add_argument("--wheels-only", action="store_true")
    p.add_argument("--wheel-mirror", default="", help="Optional package mirror; SHA-256 remains verified against PyPI")
    p.add_argument("--download-only", action="store_true")
    args = p.parse_args()
    if args.workers < 1 or args.workers > 64:
        p.error("workers must be between 1 and 64")
    args.root.mkdir(parents=True, exist_ok=True)
    if args.wheels_only:
        wheels(args)
        return
    # The OS releases this lock on exit, including interrupted downloads.
    import msvcrt
    with (args.root / "installer.lock").open("a+b") as lock:
        lock.write(b"0")
        lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Another CARLA installer is already running") from exc
        archive = download(args)
        if not args.download_only:
            extract(archive, args.root)
            # CRC validation and successful extraction make the range parts redundant.
            download_root = archive.parent.resolve()
            count = (archive.stat().st_size + 16 * 1024 * 1024 - 1) // (16 * 1024 * 1024)
            for index in range(count):
                part = archive.with_suffix(".part{:04d}".format(index))
                if part.exists():
                    if part.resolve().parent != download_root:
                        raise RuntimeError("Unexpected partial file location: " + str(part))
                    part.unlink()


if __name__ == "__main__":
    main()
