#!/usr/bin/env python3
"""Pack a standard-format data root into a data release.

Usage:
    pack_data.py ROOT OUTDIR --release data-v1 [--previous OLD/data-v0.json] [--optional KIND/NAME ...]
                 [--compression zst|gz]

Release structure (doc/DATA_FORMAT.md, Section 6):

- One immutable archive per library and library version: ``<kind>-<name>-v<N>.tar.zst`` holding
  ``libraries/<kind>/<name>/``. N is the manifest's ``version``; a changed library must get a new
  version (the packer refuses to reuse a version whose files changed).
- One base archive per release: ``base-<release>.tar.zst`` with ``opensps-data.json``, the small text
  tables in ``data/``, ``DATA_README.md`` (what to cite) and ``data_release.bib`` (the references,
  exported from NASA ADS).
- The release manifest ``<release>.json``: for every archive its library, version, size, SHA-256,
  the release tag it was uploaded to and a list of mirror URLs. With --previous, libraries whose
  version and files are unchanged point at the archive of the earlier release instead of being
  packed and uploaded again.

OUTDIR receives the new archives, the release manifest, DATA_README.md and UPLOAD.txt (the files
to attach to the release tag; see upload_release.sh). Mirrors: the GitHub release of the tag (more
can be added to MIRRORS later).
"""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fsps_data import Library, read_root_index, root_index_path  # noqa: E402
from provenance import write_data_readme  # noqa: E402

DEFAULT_OPTIONAL = ["stellar_spectra/c3k_afe+0.0"]
GITHUB_REPO = "elijahmathews/opensps-data"
# Download locations of an archive, tried in order; {tag} is the release the archive was uploaded
# to. More mirrors (e.g. a Zenodo record of a tagged release) can be appended here later.
MIRRORS = [
    "https://github.com/" + GITHUB_REPO + "/releases/download/{tag}/{file}",
]
BIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_release.bib")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def zstd_writer():
    """Factory of writable zstd streams, or None (Python >= 3.14 compression.zstd, or zstandard)."""
    try:
        from compression import zstd  # Python >= 3.14
        return lambda f: zstd.ZstdFile(f, "wb", level=10)
    except ImportError:
        pass
    try:
        import zstandard
        return lambda f: zstandard.ZstdCompressor(level=10).stream_writer(f, closefd=False)
    except ImportError:
        return None


def add_tree(tar, root, rel, arcname=None):
    """Adds root/rel recursively, in sorted order, with fixed owner, mode and time (reproducible)."""
    full = os.path.join(root, rel)
    if os.path.isdir(full):
        for name in sorted(os.listdir(full)):
            add_tree(tar, root, os.path.join(rel, name))
        return
    info = tar.gettarinfo(full, arcname=arcname or rel)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o644
    with open(full, "rb") as f:
        tar.addfile(info, f)


def pack(outdir, fname, members, compression):
    """members: (root, rel, arcname or None). Returns (bytes, sha256)."""
    path = os.path.join(outdir, fname)
    tmp = path + ".part"
    with open(tmp, "wb") as raw:
        stream = zstd_writer()(raw) if compression == "zst" else gzip.GzipFile(
            fileobj=raw, mode="wb", compresslevel=6, mtime=0)
        with tarfile.open(fileobj=stream, mode="w|", format=tarfile.PAX_FORMAT) as tar:
            for root, rel, arc in members:
                add_tree(tar, root, rel, arc)
        stream.close()
    os.replace(tmp, path)
    return os.path.getsize(path), sha256(path)


def file_hashes(lib):
    return sorted([f, h] for f, h, _ in lib.files())


def urls(tag, fname):
    return [m.format(tag=tag, file=fname) for m in MIRRORS]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("outdir")
    ap.add_argument("--release", required=True, help="release tag, e.g. data-v1")
    ap.add_argument("--previous", help="release manifest of an earlier release whose unchanged archives are reused")
    ap.add_argument("--optional", nargs="*", default=DEFAULT_OPTIONAL)
    ap.add_argument("--compression", choices=["zst", "gz"], default=None,
                    help="default: zst when available (Python >= 3.14 or the zstandard module), else gz")
    a = ap.parse_args(argv)

    comp = a.compression or ("zst" if zstd_writer() else "gz")
    if comp == "zst" and zstd_writer() is None:
        sys.exit("zstd compression needs Python >= 3.14 or the zstandard module; use --compression gz")
    ext = ".tar.zst" if comp == "zst" else ".tar.gz"
    idx = read_root_index(a.root)
    os.makedirs(a.outdir, exist_ok=True)
    prev = {}
    if a.previous:
        with open(a.previous) as f:
            for e in json.load(f)["archives"]:
                if e.get("library"):
                    prev[(e["library"], e["version"])] = e

    archives, upload, libs = [], [], []
    for entry in idx["libraries"]:
        lib = Library.open(a.root, entry["kind"], entry["name"])
        key = f"{lib.kind}/{lib.name}"
        ver = int(lib.manifest.get("version", 1))
        libs.append((lib.kind, lib.name, ver))
        problems = lib.verify()
        if problems:
            sys.exit("not packing an inconsistent library:\n  " + "\n  ".join(problems))
        old = prev.get((key, ver))
        if old is not None:
            if old["files"] != file_hashes(lib):
                sys.exit(f"{key} v{ver}: files differ from the archive in release {old['release_tag']}; "
                         "give the library a new version")
            archives.append(dict(old, optional=key in a.optional))
            print(f"  {key} v{ver}: reused from {old['release_tag']}", flush=True)
            continue
        fname = f"{lib.kind}-{lib.name}-v{ver}{ext}"
        nbytes, sha = pack(a.outdir, fname, [(a.root, entry["path"], None)], comp)
        archives.append({"library": key, "version": ver, "file": fname, "bytes": nbytes, "sha256": sha,
                         "release_tag": a.release, "urls": urls(a.release, fname),
                         "files": file_hashes(lib), "optional": key in a.optional})
        upload.append(fname)
        print(f"  {fname}: {nbytes / 1e6:.1f} MB", flush=True)

    write_data_readme(os.path.join(a.outdir, "DATA_README.md"), a.release, libs)
    base = f"base-{a.release}{ext}"
    members = [(a.root, os.path.basename(root_index_path(a.root)), None)]
    members += [(a.root, os.path.join("data", f), None) for f in sorted(os.listdir(os.path.join(a.root, "data")))]
    members.append((a.outdir, "DATA_README.md", "DATA_README.md"))
    shutil.copyfile(BIB, os.path.join(a.outdir, "data_release.bib"))
    members.append((a.outdir, "data_release.bib", "data_release.bib"))
    nbytes, sha = pack(a.outdir, base, members, comp)
    archives.insert(0, {"library": None, "version": a.release, "file": base, "bytes": nbytes, "sha256": sha,
                        "release_tag": a.release, "urls": urls(a.release, base),
                        "description": "root index, small text tables, DATA_README.md, data_release.bib",
                        "optional": False})
    upload.insert(0, base)

    rel = {"format": "opensps-data-release", "format_version": 2, "release": a.release,
           "data_format_version": idx["format_version"], "mirrors": MIRRORS, "archives": archives}
    mpath = os.path.join(a.outdir, f"{a.release}.json")
    with open(mpath, "w") as f:
        json.dump(rel, f, indent=1)
        f.write("\n")
    upload += [os.path.basename(mpath), "DATA_README.md", "data_release.bib"]
    with open(os.path.join(a.outdir, "UPLOAD.txt"), "w") as f:
        f.write("\n".join(upload) + "\n")
    new = sum(x["bytes"] for x in archives if x["release_tag"] == a.release)
    print(f"{len(archives)} archives, {len(upload) - 3} new ({new / 1e6:.0f} MB to upload); manifest {mpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
