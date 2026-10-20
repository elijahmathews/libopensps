#!/usr/bin/env python3
"""Self-contained tests of tools/fsps_data (run: python3 tools/fsps_data/test_fsps_data.py).

Synthetic inputs in a temporary directory: the format's write/read round trip, the Fortran
list-directed reader, and the converters of the upstream [alpha/Fe] layouts (alpha-enhanced
MIST isochrones, per-[alpha/Fe] C3K spectra), including an irregular grid; and the retries of
tools/fetch_data.py against a local HTTP server (no network). Without NumPy only the latter run,
and the test reports SKIP (exit status 77) if they pass.
"""
import http.server
import importlib.util
import os
import sys
import tempfile
import threading
import urllib.error

HAVE_NUMPY = importlib.util.find_spec("numpy") is not None
if HAVE_NUMPY:
    import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))  # tools/, for fetch_data
import fetch_data  # noqa: E402
if HAVE_NUMPY:
    from fsps_data import Library, LibraryWriter, ListReader, fortran_float
    from convert_legacy import Converter

FAILED = []


def check(cond, msg):
    print(("ok     " if cond else "FAILED ") + msg)
    if not cond:
        FAILED.append(msg)


def test_round_trip(tmp):
    rng = np.random.default_rng(1)
    flux = rng.random((2, 3, 4, 5)).astype("<f4")
    lib = LibraryWriter(tmp, "stellar_spectra", "toy", title="toy")
    lib.metadata["solar_metallicity"] = 0.0142
    lib.axis("Z", "mass fraction", values=[0.001, 0.0142])
    lib.axis("logg", "dex", values=[1.0, 2.0, 3.0])
    lib.axis("logT", "dex", values=[3.5, 3.6, 3.7, 3.8])
    lib.axis("wavelength", "angstrom", data=np.linspace(1000.0, 2000.0, 5))
    lib.array("flux", flux, axes=["Z", "logg", "logT", "wavelength"], dtype="<f4")
    lib.write()
    back = Library.open(tmp, "stellar_spectra", "toy")
    check(np.array_equal(back.array("flux"), flux), "round trip: FP32 array identical")
    check(back.axis_values("Z")[1] == 0.0142, "round trip: inline axis value exact")
    check(back.verify() == [], "round trip: checksums verify")
    import json
    import fsps_data as fd
    check(json.load(open(os.path.join(back.path, "manifest.json")))["format"] == "opensps-data",
          "new libraries are written as opensps-data")
    mp = os.path.join(back.path, "manifest.json")
    m = json.load(open(mp))
    m["format"] = "fsps-data"
    json.dump(m, open(mp, "w"))
    check(Library.open(tmp, "stellar_spectra", "toy").name == "toy", "transition alias: an fsps-data manifest reads")
    fd.write_root_index(tmp, "test")
    os.rename(os.path.join(tmp, "opensps-data.json"), os.path.join(tmp, "fsps-data.json"))
    check(fd.read_root_index(tmp)["release"] == "test", "transition alias: an fsps-data.json root index reads")
    ref = os.path.join(tmp, "ref.bin")
    open(ref, "wb").write(bytes(range(256)) * 3)
    w = LibraryWriter(tmp, "references", "toyref")
    w.file("ref.bin", ref)
    w.write()
    lr = Library.open(tmp, "references", "toyref")
    check(lr.verify() == [] and open(os.path.join(lr.path, "ref.bin"), "rb").read() == open(ref, "rb").read(),
          "opaque files: byte-identical copy, verified")
    with open(os.path.join(back.path, "flux.bin"), "r+b") as f:
        f.write(b"\0")
    check(back.verify() != [], "round trip: a corrupted file is detected")


def test_list_reader(tmp):
    path = os.path.join(tmp, "t.txt")
    with open(path, "w") as f:
        f.write("1 2 3 4\n\n5, 6\n7 8 9\n10\n")
    r = ListReader(path)
    check(list(r.read(5)) == [1, 2, 3, 4, 5], "list-directed read spans records")
    check(list(r.read(1)) == [7], "list-directed read discards the rest of the record")
    check(fortran_float("1.5D-3") == 1.5e-3, "D exponent")
    try:
        fortran_float("1_000")
        check(False, "reject non-Fortran numbers")
    except ValueError:
        check(True, "reject non-Fortran numbers")


def write_isoc(path, ages, npts, seed):
    rng = np.random.default_rng(seed)
    rows = []
    with open(path, "w") as f:
        for age, n in zip(ages, npts):
            f.write("# log(age) Mini Mact logl logt logg Composition Phase logMdot\n")
            for _ in range(n):
                v = [float(age)] + [float(x) for x in np.round(rng.random(8), 8)]
                rows.append(v[1:])
                f.write(" ".join(repr(x) for x in v) + "\n")
    return rows


def test_upstream_mist(tmp):
    src = os.path.join(tmp, "MIST")
    os.makedirs(src)
    with open(os.path.join(src, "zlegend.dat"), "w") as f:
        f.write("m050 -0.50\np000  0.00\n")
    expected = {}
    for ia, at in enumerate(["m2", "p0", "p4"]):
        for iz, zt in enumerate(["m050", "p000"]):
            if (at, zt) == ("p4", "m050"):
                continue  # irregular grid: one combination absent
            expected[(ia, iz)] = write_isoc(os.path.join(src, f"isoc_feh_{zt}_afe_{at}_vvcrit0.4_full.dat"),
                                            [6.0, 7.0], [3, 2 + ia], seed=10 * ia + iz)
    out = os.path.join(tmp, "out")
    Converter(tmp, out).upstream_mist(src, name="mist_afe", nt=2)
    lib = Library.open(out, "isochrones", "mist_afe")
    check(list(lib.axis_values("afe")) == [-0.2, 0.0, 0.4], "MIST afe axis from the file tags")
    check(lib.manifest["axes"]["afe"]["default"] == 0.0, "MIST afe default 0")
    check(list(lib.axis_values("feh")) == [-0.5, 0.0], "MIST [Fe/H] axis from zlegend.dat")
    avail = lib.array("available")
    check(avail.tolist() == [[1, 1], [1, 1], [0, 1]], "MIST available mask marks the absent file")
    npts = lib.array("npoints")
    mini = lib.array("mini")
    ok = True
    row = 0
    for ia in range(3):
        for iz in range(2):
            n = int(npts[ia, iz].sum())
            if (ia, iz) in expected:
                ok &= np.array_equal(mini[row:row + n], np.array(expected[(ia, iz)])[:, 0])
            else:
                ok &= n == 0
            row += n
    check(ok, "MIST columns ordered by afe, feh, isochrone, mass, values exact")


def test_upstream_c3k(tmp):
    name = "c3k_lr"
    src = os.path.join(tmp, name)
    os.makedirs(src)
    nspec, nt, ng = 7, 3, 2
    np.savetxt(os.path.join(src, name + ".lambda"), np.linspace(100.0, 1.0e4, nspec))
    np.savetxt(os.path.join(src, name + ".res"), np.full(nspec, 2.5))
    with open(os.path.join(src, name + "_zlegend.dat"), "w") as f:
        f.write("-0.25\n+0.00\n")
    with open(os.path.join(src, "logt.dat"), "w") as f:
        f.write(" 3.3979   2500\n 3.4472   2800\n 3.5000   3162\n")
    np.savetxt(os.path.join(src, "logg.dat"), [-1.0, -0.5])
    rng = np.random.default_rng(3)
    blocks = {}
    for at in ["-0.2", "+0.0", "+0.4"]:
        for zt in ["-0.25", "+0.00"]:
            a = rng.random(nspec * nt * ng).astype("<f4")
            a.tofile(os.path.join(src, f"{name}_feh{zt}_afe{at}.spec.bin"))
            blocks[(at, zt)] = a
    out = os.path.join(tmp, "out")
    Converter(tmp, out).upstream_c3k(src)
    lib = Library.open(out, "stellar_spectra", name)
    check(list(lib.axis_values("afe")) == [-0.2, 0.0, 0.4], "C3K afe axis sorted numerically")
    check(list(lib.axis_values("feh")) == [-0.25, 0.0], "C3K [Fe/H] axis keeps two decimals")
    check(list(lib.axis_values("logT")) == [3.3979, 3.4472, 3.5], "C3K log T from the first column")
    flux = lib.array("flux")
    check(flux.shape == (3, 2, ng, nt, nspec), "C3K flux shape [afe, feh, logg, logT, wavelength]")
    check(np.array_equal(flux[2, 0].ravel(), blocks[("+0.4", "-0.25")]), "C3K block bytes preserved")
    check("available" not in lib.manifest["arrays"], "C3K regular grid has no available mask")


def test_bibliography(tmp):
    import bibtex
    import provenance
    e = bibtex.parse()
    check(len(e) == 48, "data_release.bib: 48 entries parsed")
    check(all("author" in v and "year" in v and "title" in v for v in e.values()),
          "every entry has author, year and title")
    check(all(v.get("adsurl", "").endswith(k) for k, v in e.items()), "bibcodes match the ADS URLs")
    try:
        provenance.check()
        ok = True
    except KeyError:
        ok = False
    check(ok, "every bibcode of the provenance table is in data_release.bib")
    used = set(provenance.FSPS)
    for p in provenance.P.values():
        used |= set(p["cite"]) | set(p["cite_code"])
    check(used == set(e), "every entry of data_release.bib is cited by some library")
    r = bibtex.citation("2002A&A...393..167L", e)
    check(r["reference"] == "Lan\u00e7on & Mouhcine 2002, A&A, 393, 167" and r["doi"].startswith("10.1051/"),
          "references rendered from the fields (accents, journal macro, DOI)")
    check(bibtex.render(e["1992IAUS..149..225K"]) == "Kurucz 1992, IAU Symp., 149, 225",
          "proceedings rendered from series, volume and page")
    rec = provenance.provenance("isochrones", "mist")
    check([c["bibcode"] for c in rec["cite"]] == ["2016ApJ...823..102C", "2016ApJS..222....8D"]
          and not any("TODO" in c["reference"] for c in rec["cite"]), "manifest provenance cites by bibcode")


class ScriptedHandler(http.server.BaseHTTPRequestHandler):
    """Answers each GET with the next status of the server's script (200 sends BODY); counts
    the requests."""
    BODY = b"archive bytes"

    def do_GET(self):  # noqa: N802 - http.server's name
        srv = self.server
        srv.requests += 1
        code = srv.script.pop(0) if srv.script else 200
        self.send_response(code)
        if code == 429:
            self.send_header("Retry-After", "1")
        body = self.BODY if code == 200 else b"error"
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def scripted_fetch(tmp, name, script, attempts=5):
    """Fetches from a local server that answers with the statuses of `script`; returns
    (exception or None, number of requests, delays slept, file content or None)."""
    srv = http.server.HTTPServer(("127.0.0.1", 0), ScriptedHandler)
    srv.script, srv.requests = list(script), 0
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    delays, dest, err = [], os.path.join(tmp, name), None
    try:
        fetch_data.fetch(f"http://127.0.0.1:{srv.server_address[1]}/{name}", dest, attempts=attempts,
                         base_delay=1.0, sleep=delays.append)
    except Exception as exc:  # noqa: BLE001
        err = exc
    finally:
        srv.shutdown()
        srv.server_close()
    content = open(dest, "rb").read() if os.path.exists(dest) else None
    return err, srv.requests, delays, content


def test_fetch_retry(tmp):
    err, n, delays, content = scripted_fetch(tmp, "a", [500, 503, 502])
    check(err is None and n == 4 and content == ScriptedHandler.BODY, "HTTP 5xx retried until the download succeeds")
    check(len(delays) == 3 and all(0.5 * 2 ** k <= d < 1.5 * 2 ** k for k, d in enumerate(delays)),
          f"exponential backoff with jitter between attempts ({', '.join(f'{d:.2f}' for d in delays)} s)")
    err, n, delays, content = scripted_fetch(tmp, "b", [429])
    check(err is None and n == 2 and delays[0] >= 1.0, "HTTP 429 retried, honouring Retry-After")
    err, n, delays, content = scripted_fetch(tmp, "c", [404, 200])
    check(isinstance(err, urllib.error.HTTPError) and err.code == 404 and n == 1 and not delays and content is None,
          "HTTP 404 not retried")
    err, n, delays, content = scripted_fetch(tmp, "d", [500] * 9, attempts=3)
    check(isinstance(err, urllib.error.HTTPError) and err.code == 500 and n == 3 and len(delays) == 2,
          "a persistent HTTP 500 raised after the last attempt")
    # A refused connection (a port nobody listens on) is transient
    srv = http.server.HTTPServer(("127.0.0.1", 0), ScriptedHandler)
    port = srv.server_address[1]
    srv.server_close()
    delays = []
    try:
        fetch_data.fetch(f"http://127.0.0.1:{port}/e", os.path.join(tmp, "e"), attempts=3, base_delay=0.0,
                         sleep=delays.append)
        err = None
    except Exception as exc:  # noqa: BLE001
        err = exc
    check(isinstance(err, urllib.error.URLError) and len(delays) == 2, "connection errors retried")
    check(not fetch_data.is_transient(OSError("disk full")), "local I/O errors not retried")


def main():
    tests = [test_fetch_retry]
    if HAVE_NUMPY:
        tests += [test_round_trip, test_list_reader, test_upstream_mist, test_upstream_c3k, test_bibliography]
    with tempfile.TemporaryDirectory() as tmp:
        for t in tests:
            d = os.path.join(tmp, t.__name__)
            os.makedirs(d)
            t(d)
    print(f"{'FAILED: ' + str(len(FAILED)) if FAILED else 'all passed'}")
    if FAILED:
        return 1
    if not HAVE_NUMPY:
        print("NumPy not found: the tests of the format tools skipped")
        return 77
    return 0


if __name__ == "__main__":
    sys.exit(main())
