# Contributing to libopensps

Thanks for your interest in contributing to libopensps! This document covers the
essentials for getting a development environment running and submitting a change.

libopensps is a modernized FSPS: its physics must stay correct and traceable to legacy FSPS
or to independent identities. Most of the rules below exist to protect that.

## Development setup

You need:

- a Fortran 2008 compiler: gfortran (the reference compiler), Intel ifx or LLVM flang;
  NVIDIA HPC SDK (nvfortran) or LLVM flang for GPU builds (OpenMP target offload);
- Meson and Ninja (`python3 -m pip install --user meson ninja` if your system lacks them);
- Python 3 (standard library only) for the data tools.

Get the code and the data. The data libraries (stellar spectra, isochrones, nebular grids,
dust, filters, regression references) are not in this repository; they are versioned releases
of [opensps-data](https://github.com/elijahmathews/opensps-data), fetched and checksummed by:

```
git clone https://github.com/elijahmathews/libopensps.git
cd libopensps
python3 tools/fetch_data.py ~/opensps-data --release <data release named in the repository>
export OPENSPS_DATA_HOME=~/opensps-data
```

Build:

```
meson setup build --buildtype=release                              # CPU, FP64
meson setup build32 --buildtype=release -Dkernel_precision=single  # FP32 kernels
meson compile -C build
```

GPU builds (OpenMP target offload with nvfortran or LLVM flang) are
described in `doc/GPU_PLAN.md`.

## Running the test suite

```
meson test -C build --suite unit                       # no data needed
meson test -C build --suite unit --suite regression    # needs OPENSPS_DATA_HOME
```

- Run both suites in an FP64 **and** an FP32 build before opening a pull request. FP32 builds
  compare against the FP64 references with kernel-precision tolerances.
- The regression suites compare against stored reference outputs, which come with the data
  release. They change only for an intended, explained physics change (see below).
- Changes to physics are also checked against legacy FSPS on identical data
  (`tools/legacy_parity/`; it needs a legacy FSPS build and is run by hand).
- GPU or OpenMP-offload changes: compare the GPU build with the CPU build over the case matrix
  (`tools/compare_builds/`).

## Linting

No formatter or linter is enforced for the Fortran sources yet; follow the conventions below.
GitHub workflow files are checked with `actionlint`.

## Coding conventions

- **Fortran:** 2008 free form, 4-space indentation, lines of at most 132 characters (the
  compiler limit; prefer about 100). Lower-case keywords and names, `UPPER_CASE` for named
  constants.
- **Modules:** `implicit none` and `private` by default, with an explicit `public` list. Every
  dummy argument has an `intent`.
- **Precision:** use the two kinds of `src/core`: `WP` (always FP64) for setup, weights and the
  public API, `KP` (kernel precision) for per-pixel arrays. Never write `real(8)` or literal
  kinds.
- **No magic numbers:** physical constants and model parameters are named constants, with
  their source in a comment.
- **Documentation:** `!>` comments (Doxygen style) on modules and public procedures. A physics
  choice gets a comment that says why, with a reference.
- **Floating point:** never `-ffast-math`, nvfortran `-fast`, or any flag that reassociates
  floating point (it deletes the compensated sums; Intel compilers do so by default, so
  `meson.build` sets `-fp-model=precise` for them). FP32 kernels follow the rules in
  `doc/MIXED_PRECISION.md`; device code follows `doc/GPU_PLAN.md` ("Rules learned so far").
- **C API:** public functions carry the library prefix, report errors through a status
  argument, and are versioned: an addition raises the minor API version, a change or removal
  the major version, recorded in the C API documentation.
- **Match the surrounding code.** A change that touches a file follows that file's style.

## Commit conventions

Commit messages and pull request titles follow
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```
<type>(<optional scope>): <imperative summary, lower case, no final period>

<body: what changed and why, wrapped at 72 characters; numbers where they matter
(timings, maximum differences, test counts)>

<footers: BREAKING CHANGE: ..., Co-Authored-By: ..., Refs #123>
```

- **Types:** `feat` (new capability), `fix` (a bug, including a physics bug), `perf`,
  `refactor` (no change in results), `test`, `docs`, `build`, `ci`, `chore`.
- **Scopes** name the area: `ssp`, `csp`, `batch`, `imf`, `isochrones`, `dust`, `nebular`,
  `agn`, `igm`, `photometry`, `io`, `data`, `capi`, `gpu`, `build`, `ci`.
- **Breaking changes** to the C API or the data format: `!` after the type/scope and a
  `BREAKING CHANGE:` footer saying what callers must change.
- **A change in physical results** says so in the body, with its size and the reason (for
  example "spectra change by up to 2e-5 below 1500 A: the curve is now continuous there").
- Pull requests are squash-merged, so **the pull request title and description become the
  commit on `main`**: write them to these rules. Commits inside a pull request can be informal.
- Commits should be signed (OpenPGP or SSH) where possible; this is encouraged, not required.

## Submitting a change

1. Branch from `main`; keep a pull request to one logical change (small pull requests are
   reviewed and merged faster). Large structural work goes on a development branch agreed in
   an issue first.
2. Add or update tests with every change: a unit test for new code, a regression case for new
   physics paths.
3. If results change on purpose, regenerate the regression references with the documented
   commands, explain the change in the pull request, and check it against legacy FSPS where it
   applies (`tools/legacy_parity/README.md` lists the intentional differences; add yours).
   New references are published as a new version of the references library in the data
   release by a maintainer.
4. Open the pull request against `main` with a Conventional Commits title. CI must be green;
   say in the description what you ran locally (builds, suites, counts).
5. A maintainer reviews and squash-merges.

## Reporting bugs / requesting features

Please use the issue templates — they help make sure a report has what's needed to
act on it. For anything security-sensitive, see [`SECURITY.md`](SECURITY.md) instead
of opening a public issue.

A good physics bug report says what you computed (parameters, libraries, version), what you
expected and why (a reference, an identity, or legacy FSPS output), and the size of the
difference.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating,
you're expected to uphold it.
