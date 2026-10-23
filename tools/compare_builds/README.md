# Comparing builds

- `run_cases.sh` and `compare_builds.py`: every case of `tests/data/param_cases.psv` through one
  build, then two builds against each other (GPU vs CPU, FP32 vs FP64 kernels, one compiler vs
  another). The recipe is in the header of each script.
- `imf_batch_dump.f90` (built as `imf_batch_dump`) and `compare_imf_dump.py`: the batched SSP
  with the IMF free per model (`compute_csp_batch` with `ssp_op`) on 64 models with extreme IMF
  parameters (MIST + MILES, z = 0.5, nebular, IGM, AGN and dust emission). Run
  `imf_batch_dump <imf_type> <file>` for imf_type 1-4 in each build (data root in
  `OPENSPS_DATA_HOME`), then `python3 compare_imf_dump.py REFERENCE_FILE TEST_FILE`: worst
  per-pixel spectrum difference (pixels above 1e-6 of the peak), magnitude difference (bands
  within 30 mag of the brightest), mass and L_bol. The `test_csp_batch` oracle covers the same
  path against per-model SSP grids; this tool compares builds.
