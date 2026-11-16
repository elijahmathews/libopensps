#!/usr/bin/env python3
"""Pack bandpass (filter) curves into a standard-format `filters/<name>` library (doc/FILTERS_PLAN.md).

Inputs: VOTables of the SVO Filter Profile Service (https://svo2.cab.inta-csic.es/theory/fps/; TABLEDATA
serialization, standard library only), or plain two-column text files (wavelength, transmission).
The library keeps the curves as tabulated, in Angstrom, and records in its manifest, per filter,
the name, the detector type, an optional calibration, the SHA-256 content hash that libopensps
verifies on load, and where the curve came from (for an SVO profile: its ID, WavelengthRef, the
service's own zero point and detector type, kept for comparison only: libopensps computes zero
points from the Vega and solar spectra of the data release, never from a provider).

    filters_tool.py pack ROOT NAME FILE... [--detector photon|energy] [--unit angstrom|nm|um|m]
                         [--cal LABEL:LAMBDA_REF:EXPONENT] [--title T] [--version N]
    filters_tool.py svo  ROOT NAME ID... [--base-url URL]       # downloads (network), then packs

`pack` takes the label of a filter from its VOTable (SVO filterID) or file name. `--detector` and
`--unit` apply to text files (a VOTable states both); `--cal` attaches the calibration of the
legacy IR lists (exponent -1: flat nu f_nu, -2: beta = 2 blackbody; LAMBDA_REF in Angstrom).

`filter_hash` is the reference implementation of the content hash (the Fortran one is
`filter_hash` in src/spectra/opensps_filters.f90); the two are tested against each other.
"""
import argparse
import datetime
import hashlib
import os
import struct
import sys
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opensps_data import LibraryWriter, write_root_index, root_index_path  # noqa: E402

SAFE_FLOOR = 2.2250738585072014e-308   # tiny(1.0_wp): the Fortran cleaning drops a wavelength closer than this
UNIT_TO_ANGSTROM = {"angstrom": 1.0, "aa": 1.0, "a": 1.0, "nm": 10.0, "nanometer": 10.0, "um": 1.0e4,
                    "micron": 1.0e4, "micrometer": 1.0e4, "mm": 1.0e7, "m": 1.0e10}
SVO_BASE = "https://svo2.cab.inta-csic.es/theory/fps/getdata.php"


class FilterError(ValueError):
    pass


def clean(lam, resp):
    """The cleaning libopensps applies to a user curve: no NaN/Inf or wavelength <= 0; a decreasing
    curve is reversed; repeated wavelengths are dropped (the first kept); negative response is
    clamped to zero; the wavelengths must then increase strictly (at least two points)."""
    lam = np.asarray(lam, dtype=np.float64)
    resp = np.asarray(resp, dtype=np.float64)
    if lam.shape != resp.shape or lam.ndim != 1:
        raise FilterError("wavelength and response must be 1-D arrays of one length")
    if not (np.all(np.isfinite(lam)) and np.all(np.isfinite(resp))) or np.any(lam <= 0.0):
        raise FilterError("NaN/Inf, or a wavelength <= 0")
    if lam.size > 1 and lam[-1] < lam[0]:
        lam, resp = lam[::-1], resp[::-1]
    keep_l, keep_r = [], []
    for x, y in zip(lam, resp):
        if keep_l and abs(x - keep_l[-1]) < SAFE_FLOOR:
            continue
        keep_l.append(float(x))
        keep_r.append(max(float(y), 0.0))
    lc, rc = np.array(keep_l), np.array(keep_r)
    if lc.size < 2:
        raise FilterError("fewer than two distinct wavelengths")
    if np.any(np.diff(lc) <= 0.0):
        raise FilterError("wavelengths neither increasing nor decreasing")
    return lc, rc


def filter_hash(lam, resp, detector=0, calibration=None):
    """SHA-256 (hex) of the cleaned curve: tag 'OPENSPS-FILTER-v1', detector (int64), calibrated
    flag (int64), [lambda_ref, exponent] (binary64), n (int64), wavelengths, response (binary64),
    all little-endian; -0.0 is hashed as +0.0. `calibration` is (lambda_ref, exponent) or None."""
    lc, rc = clean(lam, resp)
    h = hashlib.sha256()
    h.update(b"OPENSPS-FILTER-v1")
    h.update(struct.pack("<q", int(detector)))
    h.update(struct.pack("<q", 1 if calibration else 0))
    if calibration:
        h.update(struct.pack("<dd", float(calibration[0]), float(calibration[1])))
    h.update(struct.pack("<q", lc.size))
    h.update((lc + 0.0).astype("<f8").tobytes())
    h.update((rc + 0.0).astype("<f8").tobytes())
    return h.hexdigest()


def read_ascii(path, unit="angstrom"):
    """(wavelength [A], transmission) from a text file of two columns; '#' lines and blanks skipped."""
    scale = UNIT_TO_ANGSTROM[unit.lower()]
    rows = []
    with open(path) as f:
        for line in f:
            t = line.split("#", 1)[0].split()
            if len(t) >= 2:
                rows.append((float(t[0]), float(t[1])))
    if not rows:
        raise FilterError(f"{path}: no data rows")
    a = np.array(rows)
    return a[:, 0] * scale, a[:, 1]


def read_votable(path):
    """One SVO-style VOTable (TABLEDATA): (wavelength [A], transmission, params) with the PARAM
    entries of the table (filterID, DetectorType, WavelengthRef, ZeroPoint, ...) as strings."""
    root = ET.parse(path).getroot()

    def tag(e):
        return e.tag.rsplit("}", 1)[-1]

    table = next((e for e in root.iter() if tag(e) == "TABLE"), None)
    if table is None:
        raise FilterError(f"{path}: no TABLE")
    if any(tag(e) == "BINARY" or tag(e) == "BINARY2" or tag(e) == "FITS" for e in table.iter()):
        raise FilterError(f"{path}: only the TABLEDATA serialization is supported "
                          "(SVO's default; use format=votable or ascii)")
    params = {e.get("name"): e.get("value") for e in table.iter() if tag(e) == "PARAM"}
    fields = [e for e in table.iter() if tag(e) == "FIELD"]
    names = [(e.get("name") or e.get("ID") or "").lower() for e in fields]
    try:
        iw = next(i for i, n in enumerate(names) if n.startswith("wavelength"))
        it = next(i for i, n in enumerate(names) if n.startswith("transmission"))
    except StopIteration:
        raise FilterError(f"{path}: need FIELDs Wavelength and Transmission (found {names})")
    unit = (fields[iw].get("unit") or params.get("WavelengthUnit") or "angstrom").strip().lower()
    if unit not in UNIT_TO_ANGSTROM:
        raise FilterError(f"{path}: unknown wavelength unit '{unit}'")
    tunit = (fields[it].get("unit") or "").strip()
    if tunit not in ("", "1", "dimensionless"):
        raise FilterError(f"{path}: transmission must be dimensionless (unit '{tunit}')")
    lam, trans = [], []
    for tr in (e for e in table.iter() if tag(e) == "TR"):
        td = [e.text for e in tr if tag(e) == "TD"]
        lam.append(float(td[iw]))
        trans.append(float(td[it]))
    if not lam:
        raise FilterError(f"{path}: no rows")
    return np.array(lam) * UNIT_TO_ANGSTROM[unit], np.array(trans), params


def svo_detector(params):
    """SVO's DetectorType: 0 energy counter, 1 photon counter (default photon)."""
    d = params.get("DetectorType")
    if d is None or str(d).strip() == "":
        return 0
    return 1 if int(float(d)) == 0 else 0   # libopensps: 0 photon, 1 energy


def pack(root, name, curves, title="", version=1):
    """curves: list of dicts with name, lam, resp, detector (0 photon, 1 energy), optional
    calibration (lambda_ref, exponent), description, source (dict). Writes `filters/<name>`."""
    if not curves:
        raise FilterError("no curves")
    labels = [c["name"] for c in curves]
    if len(set(labels)) != len(labels):
        raise FilterError("filter names must be unique")
    lib = LibraryWriter(root, "filters", name, title or f"Filter set {name}", version=version,
                        provenance={"converter": "tools/opensps_data/filters_tool.py"})
    lib.axis("filter", "", labels=labels)
    lib.array("npoints", np.array([len(c["lam"]) for c in curves], dtype=np.int32), axes=["filter"],
              dtype="<i4", units="count")
    lib.array("wavelength", np.concatenate([np.asarray(c["lam"], dtype=np.float64) for c in curves]),
              units="angstrom", description="concatenated filter curves, npoints[i] points for filter i")
    lib.array("transmission", np.concatenate([np.asarray(c["resp"], dtype=np.float64) for c in curves]),
              units="relative throughput (response per photon for detector 'photon', per unit energy "
                    "for 'energy')")
    info = []
    for c in curves:
        cal = c.get("calibration")
        e = {"name": c["name"], "description": c.get("description", ""),
             "detector": "energy" if c["detector"] == 1 else "photon",
             "hash": filter_hash(c["lam"], c["resp"], c["detector"], cal)}
        if cal:
            e["calibration"] = {"lambda_ref": float(cal[0]), "exponent": float(cal[1])}
        if c.get("source"):
            e["source"] = c["source"]
        info.append(e)
    lib.metadata["filters"] = info
    lib.metadata["hash_note"] = ("hash: SHA-256 of the cleaned curve, detector and calibration "
                                 "(doc/FILTERS_PLAN.md, 9a); libopensps refuses a curve that differs")
    man = lib.write()
    if not os.path.exists(root_index_path(root)):
        write_root_index(root, name)
    return man


def curve_from_file(path, detector, unit, cal=None):
    if path.lower().endswith((".xml", ".vot", ".votable")):
        lam, trans, params = read_votable(path)
        label = params.get("filterID") or os.path.splitext(os.path.basename(path))[0]
        src = {"file": os.path.basename(path)}
        for k in ("filterID", "FilterProfileService", "WavelengthRef", "WavelengthMean", "WavelengthEff",
                  "ZeroPoint", "ZeroPointType", "PhotCalID", "DetectorType", "Description"):
            if params.get(k) not in (None, ""):
                src["svo_" + k] = params[k]
        c = {"name": label, "lam": lam, "resp": trans, "detector": svo_detector(params),
             "description": params.get("Description", ""), "source": src}
    else:
        lam, trans = read_ascii(path, unit)
        label = os.path.splitext(os.path.basename(path))[0]
        c = {"name": label, "lam": lam, "resp": trans, "detector": detector,
             "source": {"file": os.path.basename(path)}}
    if cal and c["name"] in cal:
        c["calibration"] = cal[c["name"]]
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("pack", "svo"):
        p = sub.add_parser(cmd)
        p.add_argument("root")
        p.add_argument("name")
        p.add_argument("inputs", nargs="+")
        p.add_argument("--title", default="")
        p.add_argument("--version", type=int, default=1)
        p.add_argument("--cal", action="append", default=[], metavar="LABEL:LAMBDA_REF:EXPONENT")
        if cmd == "pack":
            p.add_argument("--detector", choices=("photon", "energy"), default="photon")
            p.add_argument("--unit", default="angstrom", choices=sorted(UNIT_TO_ANGSTROM))
        else:
            p.add_argument("--base-url", default=SVO_BASE)
    a = ap.parse_args(argv)
    cal = {}
    for s in a.cal:
        label, lref, expo = s.rsplit(":", 2)
        cal[label] = (float(lref), float(expo))
    files = a.inputs
    detector, unit = (1 if getattr(a, "detector", "photon") == "energy" else 0), getattr(a, "unit", "angstrom")
    if a.cmd == "svo":
        files = []
        os.makedirs(os.path.join(a.root, "svo_cache"), exist_ok=True)
        for fid in a.inputs:
            url = f"{a.base_url}?format=votable&id={fid}"
            dst = os.path.join(a.root, "svo_cache", fid.replace("/", "_") + ".xml")
            req = urllib.request.Request(url, headers={"User-Agent": "opensps-filters-tool"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
                f.write(r.read())
            files.append(dst)
        print("retrieved", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"), file=sys.stderr)
    curves = [curve_from_file(f, detector, unit, cal) for f in files]
    man = pack(a.root, a.name, curves, a.title, a.version)
    print(f"filters/{a.name}: {len(curves)} filters")
    for e in man["metadata"]["filters"]:
        print(f"  {e['name']:<28s} {e['detector']:<7s} {e['hash']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
