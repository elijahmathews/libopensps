"""Read, write and verify data libraries of libopensps in the standard format (doc/DATA_FORMAT.md).

A data root holds ``opensps-data.json`` and ``libraries/<kind>/<name>/manifest.json`` with one raw
little-endian array per ``.bin`` file. This module needs only the standard library and NumPy.

Writing a library::

    lib = LibraryWriter(root, "stellar_spectra", "mylib", title="My library")
    lib.metadata["solar_metallicity"] = 0.0142
    lib.axis("Z", "mass fraction", values=[0.001, 0.0142, 0.03])
    lib.axis("wavelength", "angstrom", data=wave)               # stored in a file
    lib.array("flux", flux, axes=["Z", "logg", "logT", "wavelength"], units="...")
    lib.write()

Reading::

    lib = Library.open(root, "stellar_spectra", "mylib")
    flux = lib.array("flux")          # numpy array, C order, shape as in the manifest
    z = lib.axis_values("Z")
"""

import datetime
import hashlib
import json
import os
import re

import numpy as np

FORMAT = "opensps-data"
FORMAT_VERSION = 1
ROOT_INDEX = "opensps-data.json"
# Names before the libopensps rename, still accepted when reading (transition alias)
OLD_FORMAT = "fsps-data"
OLD_ROOT_INDEX = "fsps-data.json"
FORMATS = (FORMAT, OLD_FORMAT)


def root_index_path(root):
    """The root index of a data root: opensps-data.json, or the older fsps-data.json."""
    p = os.path.join(root, ROOT_INDEX)
    old = os.path.join(root, OLD_ROOT_INDEX)
    return p if os.path.exists(p) or not os.path.exists(old) else old
MANIFEST = "manifest.json"
DTYPES = {"<f4": np.dtype("<f4"), "<f8": np.dtype("<f8"), "<i4": np.dtype("<i4"), "<i8": np.dtype("<i8")}

# Arrays smaller than this many values go inline into the manifest when given as axis values
INLINE_AXIS_MAX = 512


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def library_dir(root, kind, name):
    return os.path.join(root, "libraries", kind, name)


def _dtype_name(a):
    dt = np.asarray(a).dtype
    if dt.kind == "f":
        return "<f4" if dt.itemsize == 4 else "<f8"
    if dt.kind in "iu":
        return "<i4" if dt.itemsize <= 4 else "<i8"
    raise ValueError(f"unsupported dtype {dt}")


def _json_number(x):
    """A Python number that json writes so that it reads back exactly."""
    if isinstance(x, (np.integer, int)):
        return int(x)
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("non-finite number in manifest")
    return v


class LibraryWriter:
    """Builds one library directory: arrays first (written immediately), manifest last."""

    def __init__(self, root, kind, name, title="", notes="", provenance=None, version=1):
        self.root = root
        self.version = int(version)   # version of this library's data; bump on any change
        self.kind = kind
        self.name = name
        self.dir = library_dir(root, kind, name)
        os.makedirs(self.dir, exist_ok=True)
        self.title = title
        self.notes = notes
        self.provenance = dict(provenance or {})
        self.metadata = {}
        self.axes = {}
        self.arrays = {}
        self.opaque = {}

    def _write_file(self, fname, data, dtype):
        a = np.ascontiguousarray(data)
        if dtype is None:
            dtype = _dtype_name(a)
        dt = DTYPES[dtype]
        if a.dtype.kind == "f" and dt.kind == "f" and a.dtype.itemsize > dt.itemsize:
            # Narrowing is a decision of the caller, never implicit
            raise ValueError(f"{fname}: refusing to narrow {a.dtype} to {dtype}")
        out = a.astype(dt, copy=False)
        if out.dtype != a.dtype and not np.array_equal(out, a, equal_nan=(a.dtype.kind == "f")):
            raise ValueError(f"{fname}: conversion to {dtype} changed values")
        path = os.path.join(self.dir, fname)
        out.tofile(path)
        return {"file": fname, "dtype": dtype, "shape": [int(n) for n in out.shape],
                "sha256": sha256_file(path)}

    def axis(self, name, units, values=None, data=None, labels=None, dtype=None, description=None, **extra):
        """An axis with inline `values`, file-backed `data`, or string `labels`."""
        ax = {"units": units}
        if description:
            ax["description"] = description
        ax.update(extra)
        if labels is not None:
            ax["labels"] = [str(s) for s in labels]
        elif values is not None and data is None and len(values) <= INLINE_AXIS_MAX:
            ax["values"] = [_json_number(v) for v in values]
        else:
            arr = np.asarray(data if data is not None else values)
            if arr.ndim != 1:
                raise ValueError(f"axis {name} must be 1-D")
            ax.update(self._write_file(f"axis_{name}.bin", arr, dtype))
        self.axes[name] = ax

    def array(self, name, data, axes=None, units="", dtype=None, description=None, **extra):
        desc = self._write_file(f"{name}.bin", data, dtype)
        if axes is not None:
            if len(axes) != len(desc["shape"]):
                raise ValueError(f"array {name}: {len(axes)} axes for shape {desc['shape']}")
            for ax, n in zip(axes, desc["shape"]):
                if ax not in self.axes:
                    raise ValueError(f"array {name}: unknown axis {ax}")
                alen = axis_length(self.axes[ax], self.dir)
                if alen != n:
                    raise ValueError(f"array {name}: axis {ax} has {alen} values, dimension is {n}")
            desc["axes"] = list(axes)
        desc["units"] = units
        if description:
            desc["description"] = description
        desc.update(extra)
        # Keep a stable key order: file, dtype, shape, axes, units, ..., sha256 last
        sha = desc.pop("sha256")
        desc["sha256"] = sha
        self.arrays[name] = desc

    def file(self, name, src, description=""):
        """An opaque file (e.g. a regression reference), copied byte for byte from `src`."""
        import shutil
        fname = os.path.basename(src)
        dst = os.path.join(self.dir, fname)
        shutil.copyfile(src, dst)
        desc = {"file": fname, "bytes": os.path.getsize(dst)}
        if description:
            desc["description"] = description
        desc["sha256"] = sha256_file(dst)
        self.opaque[name] = desc

    def write(self):
        prov = {"converter": "tools/fsps_data", "created":
                datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        prov.update(self.provenance)
        man = {"format": FORMAT, "format_version": FORMAT_VERSION, "kind": self.kind,
               "name": self.name, "version": self.version}
        if self.title:
            man["title"] = self.title
        if self.notes:
            man["notes"] = self.notes
        man["provenance"] = prov
        man["metadata"] = {k: (_json_number(v) if isinstance(v, (int, float, np.number)) else v)
                           for k, v in self.metadata.items()}
        man["axes"] = self.axes
        man["arrays"] = self.arrays
        if self.opaque:
            man["files"] = self.opaque
        with open(os.path.join(self.dir, MANIFEST), "w") as f:
            json.dump(man, f, indent=1)
            f.write("\n")
        return man


def axis_length(ax, libdir=None):
    if "labels" in ax:
        return len(ax["labels"])
    if "values" in ax:
        return len(ax["values"])
    return int(np.prod(ax["shape"]))


class Library:
    """A library read back from disk (manifest plus lazily loaded arrays)."""

    def __init__(self, path, manifest):
        self.path = path
        self.manifest = manifest
        self.kind = manifest["kind"]
        self.name = manifest["name"]
        self.metadata = manifest.get("metadata", {})

    @classmethod
    def open(cls, root, kind=None, name=None):
        path = root if kind is None else library_dir(root, kind, name)
        with open(os.path.join(path, MANIFEST)) as f:
            man = json.load(f)
        if man.get("format") not in FORMATS or man.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{path}: not an {FORMAT} v{FORMAT_VERSION} library")
        return cls(path, man)

    def _load(self, desc, mmap=False):
        dt = DTYPES[desc["dtype"]]
        fpath = os.path.join(self.path, desc["file"])
        n = int(np.prod(desc["shape"]))
        size = os.path.getsize(fpath)
        if size != n * dt.itemsize:
            raise ValueError(f"{fpath}: {size} bytes, expected {n * dt.itemsize}")
        if mmap:
            return np.memmap(fpath, dtype=dt, mode="r", shape=tuple(desc["shape"]))
        return np.fromfile(fpath, dtype=dt).reshape(desc["shape"])

    def array(self, name, mmap=False):
        return self._load(self.manifest["arrays"][name], mmap)

    def axis_values(self, name):
        ax = self.manifest["axes"][name]
        if "labels" in ax:
            return list(ax["labels"])
        if "values" in ax:
            return np.asarray(ax["values"], dtype=np.float64)
        return self._load(ax)

    def files(self):
        """(file name, sha256, expected bytes) of every file the manifest references."""
        out = []
        for desc in list(self.manifest["axes"].values()) + list(self.manifest["arrays"].values()):
            if "file" in desc:
                n = int(np.prod(desc["shape"])) * DTYPES[desc["dtype"]].itemsize
                out.append((desc["file"], desc["sha256"], n))
        for desc in self.manifest.get("files", {}).values():
            out.append((desc["file"], desc["sha256"], int(desc["bytes"])))
        return out

    def verify(self):
        """List of problems (empty when every file has its size and checksum)."""
        problems = []
        for fname, sha, nbytes in self.files():
            fpath = os.path.join(self.path, fname)
            if not os.path.exists(fpath):
                problems.append(f"{fpath}: missing")
                continue
            if os.path.getsize(fpath) != nbytes:
                problems.append(f"{fpath}: {os.path.getsize(fpath)} bytes, expected {nbytes}")
                continue
            if sha256_file(fpath) != sha:
                problems.append(f"{fpath}: checksum mismatch")
        return problems


def write_root_index(root, release, title=""):
    """Writes opensps-data.json listing every library found under root/libraries."""
    libs = []
    base = os.path.join(root, "libraries")
    for kind in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        for name in sorted(os.listdir(os.path.join(base, kind))):
            if os.path.exists(os.path.join(base, kind, name, MANIFEST)):
                libs.append({"kind": kind, "name": name, "path": f"libraries/{kind}/{name}"})
    idx = {"format": FORMAT, "format_version": FORMAT_VERSION, "release": release}
    if title:
        idx["title"] = title
    idx["libraries"] = libs
    with open(os.path.join(root, ROOT_INDEX), "w") as f:
        json.dump(idx, f, indent=1)
        f.write("\n")
    return idx


def read_root_index(root):
    with open(root_index_path(root)) as f:
        idx = json.load(f)
    if idx.get("format") not in FORMATS or idx.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{root}: not an {FORMAT} v{FORMAT_VERSION} data root")
    return idx


# ---------------------------------------------------------------------------------------------
# Reading legacy text exactly as Fortran list-directed input does
# ---------------------------------------------------------------------------------------------

_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+)?$")


def fortran_float(tok):
    """A real as Fortran reads it list-directed (correctly rounded, like Python's float)."""
    if not _NUMBER.match(tok):
        raise ValueError(f"not a Fortran real: {tok!r}")
    return float(tok.replace("d", "e").replace("D", "e"))


def split_tokens(line):
    """Value separators of list-directed input: blanks, tabs and commas."""
    return [t for t in re.split(r"[\s,]+", line.strip()) if t]


class ListReader:
    """Sequential list-directed reads over the records (lines) of a text file.

    ``read(n)`` reads n values starting at the next record, continuing over as many records as
    needed, and discards the rest of the last record, as ``read(u, *) a(1:n)`` does. ``skip()``
    is ``read(u, *)`` with an empty list.
    """

    def __init__(self, path):
        self.path = path
        with open(path) as f:
            self.lines = f.read().splitlines()
        self.pos = 0

    def at_end(self):
        return self.pos >= len(self.lines)

    def skip(self, n=1):
        if self.pos + n > len(self.lines):
            raise EOFError(f"{self.path}: unexpected end of file")
        self.pos += n

    def read_tokens(self, n):
        out = []
        while len(out) < n:
            if self.pos >= len(self.lines):
                raise EOFError(f"{self.path}: end of file after {len(out)} of {n} values")
            toks = split_tokens(self.lines[self.pos])
            self.pos += 1
            for t in toks:
                if t == "/" or "*" in t:
                    raise ValueError(f"{self.path}:{self.pos}: unsupported list-directed token {t!r}")
            out.extend(toks[: n - len(out)])
        return out

    def read(self, n):
        return np.array([fortran_float(t) for t in self.read_tokens(n)], dtype=np.float64)

    def read_int(self, n=1):
        toks = self.read_tokens(n)
        return [int(t) for t in toks]
