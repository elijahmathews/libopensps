"""Provenance of the libraries converted from the FSPS data tree.

Citations are bibcodes of tools/fsps_data/data_release.bib, an export from NASA ADS; the
references written into the manifests and DATA_README.md are rendered from that file
(tools/fsps_data/bibtex.py), never typed here. Descriptions of the data come from the FSPS
sources (README, manual, revision history, data headers and READMEs, code comments) and from the
maintainers' notes. The redistribution note records only that FSPS has distributed these files
publicly; it is not a license grant.

Each entry: source, version (if known), notes, cite (bibcodes of the data), cite_code (bibcodes of
codes the data were computed with), todo.
"""
from bibtex import citation, parse

FSPS = ["2009ApJ...699..486C", "2010ApJ...712..833C"]
FSPS_URL = "https://github.com/cconroy20/fsps"
REDISTRIBUTION = ("Distributed publicly with FSPS (" + FSPS_URL + "), from whose data tree this library was "
                  "converted. No separate license is stated in the FSPS distribution; the terms of the original "
                  "data apply.")
TODO_LICENSE = "license or redistribution terms of the original data (not stated in the FSPS sources)"
CLOUDY = ["2013RMxAA..49..137F"]
CMFGEN = ["1987ApJS...63..947H", "1990A&A...231..116H", "1998ApJ...496..407H", "1999ApJ...519..354H",
          "2003ASPC..288..199H", "2005AJ....129..454B", "2012IAUS..282..229H", "2012MNRAS.424..252H"]

P = {}


def _e(key, source, cite, version=None, notes="", cite_code=(), todo=(), redistribution=None):
    P[key] = {"source": source, "version": version, "notes": notes, "cite": list(cite),
              "cite_code": list(cite_code), "redistribution": redistribution,
              "todo": list(todo) + ([] if redistribution else [TODO_LICENSE])}


# --- stellar spectra
_e("stellar_spectra/miles", "MILES empirical stellar library, combined with the BaSeL library outside the MILES "
   "wavelength range and stellar-parameter coverage (doc/REVISION_HISTORY)",
   ["2006MNRAS.371..703S", "2011A&A...532A..95F"],
   notes="Outside the MILES coverage the spectra are BaSeL (cite those references too).")
_e("stellar_spectra/basel", "BaSeL 3.1 theoretical library, WLBC version (file names basel_wlbc_*)",
   ["1997A&AS..125..229L", "1998A&AS..130...65L", "2002A&A...381..524W"], version="3.1 (WLBC)")
_e("stellar_spectra/c3k_afe+0.0", "C3K theoretical spectra, [alpha/Fe] = 0, downsampled from full C3K resolution and "
   "interpolated onto the BaSeL log g - log T grid (data/spectra/C3K/readme.md)",
   ["2022MNRAS.512.5329B", "2025ApJ...994..165P"],
   notes="C3K (Conroy et al., unpublished; computed with ATLAS12/SYNTHE), distributed with FSPS; no dedicated "
         "publication. Described in use by Byrne et al. 2022; alpha-enhanced extension: Park et al. 2025.")
# --- isochrones
_e("isochrones/mist", "MIST isochrones (solar metallicity 0.0142)", ["2016ApJ...823..102C", "2016ApJS..222....8D"])
_e("isochrones/pdva", "Padova 2007 isochrones (data/isochrones/Padova/Padova2007; solar metallicity 0.019)",
   ["2000A&AS..141..371G", "2007A&A...469..239M", "2008A&A...482..883M"])
_e("isochrones/prsc", "PARSEC isochrones (solar metallicity 0.01524)", ["2012MNRAS.427..127B"])
_e("isochrones/bsti", "BaSTI isochrones (solar metallicity 0.020)", ["2004ApJ...612..168P", "2007AJ....133..468C"])
_e("isochrones/gnva", "Geneva isochrones, 'older Geneva models' (doc/REVISION_HISTORY; solar metallicity 0.020)",
   ["1992A&AS...96..269S", "2001A&A...366..538L"])
_e("isochrones/mist_afe", "MIST isochrones with [alpha/Fe] variations (upstream FSPS layout isoc_feh_*_afe_*)",
   ["2016ApJ...823..102C", "2016ApJS..222....8D", "2025ApJ...994..165P"])
_e("ssp/bpss", "BPASS binary population SSPs, '-bin-imf135all_100' models; lowest metallicity omitted, ages cut at "
   "15.8 Gyr (FSPS manual, section 1.16)", ["2017PASA...34...58E", "2018MNRAS.479...75S"],
   version="BPASS v2.2 (Salpeter IMF to 100 Msun; file bpass_v2.2_salpeter100.ssp.bin)")
# --- nebular
for _iso in ("mist", "pdva", "prsc", "bpss"):
    for _d, _dn in (("dust", "with"), ("nodust", "without")):
        _e(f"nebular/{_iso}_{_d}", f"Cloudy photoionization tables provided by Nell Byler, ionized by the {_iso} "
           f"SSPs, {_dn} dust in the ionized gas (FSPS manual, section 1.13)",
           ["2017ApJ...840...44B"], cite_code=CLOUDY)
for _d, _dn in (("dust", "with"), ("nodust", "without")):
    _e(f"nebular/bpss_xray_{_d}", "Cloudy tables for BPASS SSPs with X-ray binaries in the ionizing spectrum, "
       f"{_dn} dust in the ionized gas (files ZAU_WX_*)", ["2024ApJ...960...13G"], cite_code=CLOUDY,
       notes="Data from K. Garofali, added to FSPS in 2023.")
_e("xrb/xsp", "X-ray binary spectra of simple populations (11 [Fe/H] x 10 ages)", ["2024ApJ...960...13G"],
   notes="Data from K. Garofali, added to FSPS in 2023.")
# --- dust
_e("dust_emission/DL07", "Dust emission models, Milky Way grain mix R_V = 3.1 (files DL07_MW3.1_*; header: L_nu per "
   "H nucleon, 3.3 um PAH feature reduced by 50%)", ["2007ApJ...657..810D"], version="MW3.1")
_e("dust_emission/THEMIS", "THEMIS dust emission model (files THEMIS_MW3.1_*)",
   ["2013A&A...558A..62J", "2017A&A...602A..46J"])
_e("agn_torus/clumpy", "Clumpy AGN torus models, 'reformatted for FSPS' (file header of "
   "Nenkova08_y010_torusg_n10_q2.0.dat): Y = 10, sigma = 10 (torusg), N0 = 10, q = 2.0; a small subset of the grid",
   ["2008ApJ...685..147N", "2008ApJ...685..160N"])
_e("attenuation/dirty", "Radiative-transfer attenuation curves for dusty galactic environments, MW and SMC grains, "
   "dusty/shell/cloudy geometries, homogeneous and clumpy media (files alldirty_{h,c}.dat)", ["2000ApJ...528..799W"])
_e("attenuation/smc_bar", "SMC bar extinction curve (file Gordon03_table4.dat, table 4 of the source)",
   ["2003ApJ...594..279G"])
for _c, _cn in (("orich", "O-rich"), ("crich", "C-rich")):
    _e(f"dusty_agb/{_c}", f"Circumstellar dust shells of {_cn} AGB stars computed with the radiative transfer code "
       "DUSTY (FSPS manual, section 1.11)", ["2015ApJ...806...82V"])
# --- hot stars, AGB
_e("hot_star_spectra/wmbasic", "WMBasic O-star spectral grid", [], cite_code=["2001A&A...375..161P"],
   notes="Grid courtesy of J. J. Eldridge, M. Ng and G. Taylor (as distributed with FSPS); WM-basic code: "
         "see cite_code.")
for _w, _wn in (("wn", "nitrogen"), ("wc", "carbon")):
    _e(f"hot_star_spectra/cmfgen_{_w}", f"CMFGEN Wolf-Rayet spectra, {_wn} sequence", ["2002MNRAS.337.1309S"],
       cite_code=CMFGEN, notes="Dataset: the O and Wolf-Rayet ionizing-flux grid; the WR models were computed with "
       "CMFGEN (code references in cite_code).")
_e("hot_star_spectra/post_agb", "Synthetic hydrogen-deficient post-AGB spectra, halo and solar compositions",
   ["2003A&A...403..709R"])
for _c, _cn in (("orich", "O-rich"), ("crich", "C-rich")):
    _e(f"agb_spectra/{_c}", f"Averaged empirical {_cn} TP-AGB spectra (data/spectra/AGB_spectra/README), extrapolated "
       "as described in the FSPS manual (section 1.3)", ["2002A&A...393..167L", "2000A&AS..146..217L"],
       notes="The second reference is for the extrapolation of the spectra.")
_e("agb_spectra/crich_aringer", "Synthetic carbon-star spectra used to extrapolate the C-rich TP-AGB spectra redward "
   "(FSPS manual, section 1.3)", ["2009A&A...503..913A"])
# --- calibration
_e("filters/allfilters", "Filter transmission curves of data/allfilters.dat; the source of each curve is in its "
   "'description' (from data/FILTER_LIST)", [], version="159 filters",
   notes="Cite the sources of the individual filter curves (per-filter descriptions).")
_e("reference_spectra/vega", "Model atmosphere for Vega, Teff = 9400 K, log g = 3.90, [M/H] = -0.50 (file header of "
   "A0V_KURUCZ_92.SED)", ["1992IAUS..149..225K"])
_e("reference_spectra/sun", "Solar spectrum, f_nu at 10 pc (file SUN_STScI.SED)", ["1996AJ....112..307C"],
   notes="Likely source (STScI absolute solar spectrum; not stated in the file).")

# --- regression references (generated by libopensps itself)
GENERATED = ("Generated by libopensps (https://github.com/elijahmathews/libopensps), a modernized FSPS, from the "
             "libraries of this data release; distributed under the license of libopensps.")
_e("references/sps", "Regression references of tests/test_runner.f90: SSP and CSP outputs (spectra, magnitudes, "
   "indices) for five library combinations, in the legacy binary layout of tests/generate_test_data.f90",
   [], notes="Regenerate with tests/generate_test_data.sh (writes tests/data/sps_ref_*.bin), then "
   "tools/fsps_data/make_references.py; only for an intended, explained physics change.",
   redistribution=GENERATED)
_e("references/param", "Regression references of tests/test_param_regression.f90: one .opsf file (SSP and CSP "
   "outputs) per case of tests/data/param_cases.psv", [],
   notes="Regenerate with 'test_param_regression tests/data/param_cases.psv <dir> --generate', then "
   "tools/fsps_data/make_references.py; only for an intended, explained physics change.",
   redistribution=GENERATED)

_BIB = None


def _bib():
    global _BIB
    if _BIB is None:
        _BIB = parse()
    return _BIB


def check():
    """Every bibcode of the table exists in data_release.bib (raises KeyError otherwise)."""
    for p in P.values():
        for b in p["cite"] + p["cite_code"]:
            citation(b, _bib())
    for b in FSPS:
        citation(b, _bib())


def _key(kind, name):
    key = f"{kind}/{name}"
    if key not in P and kind == "stellar_spectra" and name.startswith("c3k"):
        key = "stellar_spectra/c3k_afe+0.0"
    if key not in P:
        raise KeyError(f"no provenance for {kind}/{name}: add it to tools/fsps_data/provenance.py")
    return key


def provenance(kind, name):
    """The provenance record written into a manifest (references rendered from the bib file)."""
    p = P[_key(kind, name)]
    rec = {"source": p["source"]}
    if p["version"]:
        rec["version"] = p["version"]
    if p["notes"]:
        rec["notes"] = p["notes"]
    rec["cite"] = [citation(b, _bib()) for b in p["cite"]]
    if p["cite_code"]:
        rec["cite_code"] = [citation(b, _bib()) for b in p["cite_code"]]
    rec["cite_also"] = [citation(b, _bib()) for b in FSPS]
    rec["bibliography"] = "data_release.bib (NASA ADS export, shipped with the data release)"
    rec["redistribution"] = p["redistribution"] or REDISTRIBUTION
    rec["todo"] = list(p["todo"])
    return rec


def _refs(bibcodes):
    return "<br>".join(f"{citation(b, _bib())['reference']} ({b})" for b in bibcodes)


def write_data_readme(path, release, libraries, layout_note=""):
    """Writes the release's DATA_README.md: what each library is and what to cite."""
    lines = [
        f"# libopensps data release {release}",
        "",
        "Libraries for libopensps (https://github.com/elijahmathews/libopensps), a modernized FSPS, in its standard "
        "data format (doc/DATA_FORMAT.md of libopensps). Unpack the archives into one directory "
        "(`tools/fetch_data.py ~/opensps-data --release " + release + "` does this and verifies every checksum) and "
        "point `OPENSPS_DATA_HOME` at it.",
        "",
        "## What to cite",
        "",
        "Any use: the FSPS papers, " + "; ".join(f"{citation(b, _bib())['reference']} ({b})" for b in FSPS) + ". "
        "In addition, cite the sources of the libraries you use (below). Bibcodes refer to `data_release.bib` "
        "(an export from NASA ADS, shipped in this release); every library's `manifest.json` lists the same "
        "references under `provenance`.",
        "",
        "Redistribution: these files have been distributed publicly with FSPS (" + FSPS_URL + "). No separate "
        "license is stated there; the terms of the original data apply.",
        "",
        "| Library | Source | Cite | Codes used |",
        "|---|---|---|---|",
    ]
    for kind, name, ver in libraries:
        p = P[_key(kind, name)]
        src = p["source"] + (f". Version: {p['version']}." if p["version"] else ".") + \
            (f" {p['notes']}" if p["notes"] else "")
        lines.append(f"| `{kind}/{name}` (v{ver}) | {src} | {_refs(p['cite']) or '-'} | "
                     f"{_refs(p['cite_code']) or '-'} |")
    lines += ["", "## Layout", "", layout_note or (
        "Each library is one archive, `<kind>-<name>-v<N>.tar.zst` (or `.tar.gz`), holding "
        "`libraries/<kind>/<name>/` (a `manifest.json` and raw little-endian arrays); the base archive holds "
        "`opensps-data.json`, the small text tables in `data/`, this file and `data_release.bib`. Archives are "
        "immutable: a changed library gets a new version and a new archive, and the release manifest "
        "`<release>.json` lists, for every library, its version, SHA-256, size and download URLs (mirrors), possibly "
        "pointing at archives of an earlier release."),
        "", "## Open questions", "",
        "- For every library converted from the FSPS data tree: " + TODO_LICENSE + ".", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
