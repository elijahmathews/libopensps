#!/usr/bin/env python3
"""Download, verify and unpack a libopensps data release (doc/DATA_FORMAT.md, Section 6).

Usage:
    tools/fetch_data.py DEST [--release data-v1 | --release-manifest PATH_OR_URL]
                             [--mirror DIR_OR_URL ...] [--with KIND/NAME ...] [--all]
                             [--cache DIR] [--list]

Reads the release manifest (``<release>.json``, written by tools/fsps_data/pack_data.py; by
default the one attached to the release tag on GitHub) and fetches every archive that is not
optional (plus those named with --with, or all with --all) into a cache, trying the --mirror
locations first (a directory or base URL holding the archive files; a local manifest's own
directory is one), then each archive's mirror URLs. It checks each archive's SHA-256, unpacks it
into DEST, and checks the size and SHA-256 of every file the unpacked manifests list. Archives
already in the cache with the right checksum are not fetched again (CI caches that directory).
A download that fails transiently (HTTP 5xx or 429, a connection error or a timeout) is retried
with exponential backoff and jitter, up to 5 attempts per URL, before the next mirror is tried.
Then point OPENSPS_DATA_HOME at DEST (e.g. ~/opensps-data).

Standard library only (Python >= 3.8); .tar.zst archives need Python >= 3.14, the zstandard
module or the zstd command.
"""
import argparse
import hashlib
import http.client
import io
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

GITHUB_REPO = "elijahmathews/opensps-data"
MANIFEST_URL = "https://github.com/" + GITHUB_REPO + "/releases/download/{tag}/{tag}.json"
DEFAULT_RELEASE = "data-v2"
CHUNK = 1 << 22
# Retries of a URL after a transient failure: attempts in all, and the first delay in seconds,
# doubled at each retry and multiplied by a random factor in [0.5, 1.5) (jitter, so that parallel
# CI jobs do not retry in step); a server's Retry-After is honoured up to RETRY_MAX_DELAY
RETRY_ATTEMPTS = 5
RETRY_BASE_DELAY = 2.0
RETRY_MAX_DELAY = 60.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def is_url(s):
    return "://" in s and not s.startswith("file://")


def is_transient(exc):
    """Whether a failed download is worth retrying: HTTP 5xx and 429 (too many requests),
    connection errors and timeouts; not other HTTP 4xx (the URL is wrong or forbidden)."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500 or exc.code == 429
    if isinstance(exc, urllib.error.URLError):  # no response: refused, reset, DNS, timeout
        return True
    return isinstance(exc, (ConnectionError, TimeoutError, socket.timeout, http.client.HTTPException))


def retry_delay(attempt, exc, base_delay):
    """Seconds to wait before retry number `attempt` (1, 2, ...)."""
    delay = base_delay * 2 ** (attempt - 1) * random.uniform(0.5, 1.5)
    retry_after = exc.headers.get("Retry-After") if isinstance(exc, urllib.error.HTTPError) and exc.headers else None
    if retry_after and retry_after.strip().isdigit():
        delay = max(delay, float(retry_after))
    return min(delay, RETRY_MAX_DELAY)


def download(url, tmp):
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, CHUNK)


def fetch(src, dest, attempts=None, base_delay=None, sleep=time.sleep):
    """Copies a local path or downloads a URL to dest (through a temporary file). A download
    that fails transiently (is_transient) is retried, up to `attempts` in all, with exponential
    backoff and jitter; other failures, and the last one, are raised."""
    attempts = RETRY_ATTEMPTS if attempts is None else attempts
    base_delay = RETRY_BASE_DELAY if base_delay is None else base_delay
    tmp = dest + ".part"
    if not is_url(src):
        shutil.copyfile(src.replace("file://", ""), tmp)
    else:
        for attempt in range(1, attempts + 1):
            try:
                download(src, tmp)
                break
            except Exception as exc:  # noqa: BLE001 - classified by is_transient
                if attempt == attempts or not is_transient(exc):
                    raise
                delay = retry_delay(attempt, exc, base_delay)
                print(f"  attempt {attempt} of {attempts} failed ({exc}); retrying in {delay:.1f} s", flush=True)
                sleep(delay)
    os.replace(tmp, dest)


def open_tar(path):
    """A tar stream over a .tar.gz or .tar.zst archive."""
    if path.endswith(".tar.gz"):
        return tarfile.open(path, "r|gz")
    if path.endswith(".tar.zst"):
        try:
            from compression import zstd  # Python >= 3.14
            return tarfile.open(fileobj=zstd.ZstdFile(path, "rb"), mode="r|")
        except ImportError:
            pass
        try:
            import zstandard
            return tarfile.open(fileobj=zstandard.ZstdDecompressor().stream_reader(open(path, "rb")), mode="r|")
        except ImportError:
            pass
        if shutil.which("zstd"):
            data = subprocess.run(["zstd", "-dc", path], check=True, capture_output=True).stdout
            return tarfile.open(fileobj=io.BytesIO(data), mode="r:")
        sys.exit(f"{path}: .tar.zst needs Python >= 3.14, the zstandard module or the zstd command")
    raise ValueError(f"unknown archive type: {path}")


def safe_extract(tar, dest):
    """Unpacks only regular files and directories below dest (no links, no absolute paths)."""
    root = os.path.realpath(dest)
    for m in tar:
        target = os.path.realpath(os.path.join(dest, m.name))
        if not (m.isfile() or m.isdir()) or os.path.isabs(m.name) or \
                os.path.commonpath([root, target]) != root:
            raise ValueError(f"refusing archive member {m.name!r}")
        tar.extract(m, dest)


def verify_root(dest):
    """Problems found in the files of every library unpacked in DEST."""
    problems = []
    index = os.path.join(dest, "opensps-data.json")
    if not os.path.exists(index):
        index = os.path.join(dest, "fsps-data.json")  # name before the libopensps rename
    with open(index) as f:
        idx = json.load(f)
    for lib in idx["libraries"]:
        d = os.path.join(dest, lib["path"])
        mpath = os.path.join(d, "manifest.json")
        if not os.path.exists(mpath):
            continue  # an optional library that was not fetched
        with open(mpath) as f:
            man = json.load(f)
        for desc in list(man["axes"].values()) + list(man["arrays"].values()) + \
                list(man.get("files", {}).values()):
            if "file" not in desc:
                continue
            if "bytes" in desc:          # an opaque file (e.g. a regression reference)
                nbytes = int(desc["bytes"])
            else:
                nbytes = int(desc["dtype"][2:])
                for s in desc["shape"]:
                    nbytes *= s
            p = os.path.join(d, desc["file"])
            if not os.path.exists(p) or os.path.getsize(p) != nbytes:
                problems.append(f"{p}: missing or wrong size")
            elif sha256(p) != desc["sha256"]:
                problems.append(f"{p}: checksum mismatch")
    return problems


def candidates(entry, mirrors):
    out = [m.rstrip("/") + "/" + entry["file"] if is_url(m) else os.path.join(m, entry["file"]) for m in mirrors]
    return out + [u for u in entry.get("urls", []) if "TODO" not in u]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dest")
    ap.add_argument("--release", default=DEFAULT_RELEASE, help="release tag (default %(default)s)")
    ap.add_argument("--release-manifest", help="path or URL of the release manifest (default: from the tag)")
    ap.add_argument("--mirror", action="append", default=[], help="directory or base URL tried first")
    ap.add_argument("--with", dest="extra", nargs="*", default=[], help="optional libraries to fetch too")
    ap.add_argument("--all", action="store_true", help="fetch every archive, optional ones included")
    ap.add_argument("--cache", default=os.path.join(os.path.expanduser("~"), ".cache", "opensps-data"))
    ap.add_argument("--list", action="store_true", help="list the archives of the release and exit")
    a = ap.parse_args(argv)

    os.makedirs(a.cache, exist_ok=True)
    src = a.release_manifest or MANIFEST_URL.format(tag=a.release)
    mpath = os.path.join(a.cache, os.path.basename(src))
    try:
        fetch(src, mpath)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"cannot get the release manifest {src}: {exc}")
    with open(mpath) as f:
        rel = json.load(f)
    # 'fsps-data-release': the name before the libopensps rename, still accepted
    if rel.get("format") not in ("opensps-data-release", "fsps-data-release") or \
            rel.get("format_version") != 2 or rel.get("data_format_version") != 1:
        sys.exit(f"{src}: not a release manifest this script reads (release format 2, data format 1)")
    mirrors = list(a.mirror)
    if a.release_manifest and not is_url(a.release_manifest):
        mirrors.append(os.path.dirname(os.path.abspath(a.release_manifest.replace("file://", ""))))
    if a.list:
        for e in rel["archives"]:
            print(f"{e['file']:55s} {e['bytes'] / 1e6:8.1f} MB  {e['release_tag']}"
                  f"{'  (optional)' if e['optional'] else ''}")
        return 0

    want = [e for e in rel["archives"] if a.all or not e["optional"] or e.get("library") in a.extra]
    os.makedirs(a.dest, exist_ok=True)
    for e in want:
        cached = os.path.join(a.cache, e["file"])
        if not (os.path.exists(cached) and os.path.getsize(cached) == e["bytes"] and sha256(cached) == e["sha256"]):
            for url in candidates(e, mirrors):
                try:
                    print(f"fetching {e['file']} ({e['bytes'] / 1e6:.1f} MB) from {url}", flush=True)
                    fetch(url, cached)
                except Exception as exc:  # noqa: BLE001 - try the next mirror
                    print(f"  failed: {exc}", flush=True)
                    continue
                if sha256(cached) == e["sha256"]:
                    break
                print("  checksum mismatch", flush=True)
                os.remove(cached)
            else:
                sys.exit(f"{e['file']}: no mirror provided a valid copy")
        with open_tar(cached) as tar:
            safe_extract(tar, a.dest)
    problems = verify_root(a.dest)
    if problems:
        sys.exit("verification failed:\n  " + "\n  ".join(problems))
    print(f"release {rel['release']}: {len(want)} archives unpacked and verified in {a.dest}")
    print(f"export OPENSPS_DATA_HOME={os.path.abspath(a.dest)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
