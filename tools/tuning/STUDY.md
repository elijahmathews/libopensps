# Tuning sensitivity study: protocol for one device

Goal: for ONE device (GPU + its host CPU), measure how much each run-time tuning parameter of
`src/core/opensps_tuning.f90` moves the performance of the batched CSP, and where its best value is.
The results decide which parameters deserve a profile entry and are the data the autotuner will later
produce automatically. **Measure only. Do not edit the registry, add profile entries or change any
source file**: an entry needs a measurement cited in `doc/TUNING.md` ("Measured device data"), and
the maintainer adds it after reviewing yours.

Rules: nothing else may run on the node during a sweep (a concurrent job changes the host-side
paths); one benchmark process at a time; `OMP_TARGET_OFFLOAD=MANDATORY`; do not use the machine's
display GPU for anything but what is asked here.

## 0. Get the code and the data

```sh
git clone https://github.com/elijahmathews/libopensps.git && cd libopensps
git switch study/tuning-sensitivity          # has tools/tuning/sweep.py and the tuning registry
python3 tools/fetch_data.py $HOME/opensps-data --release data-v4
export OPENSPS_DATA_HOME=$HOME/opensps-data
```

## 1. Build twice: FP32 and FP64 kernels, the project's recommended recipe for the device

NVIDIA (P100 is `cc60`, A100 `cc80`; HPC SDK 25.1, `module load nvhpc/25.1` on the ICDS cluster):

```sh
FC=nvfortran CC=nvc meson setup b32 --buildtype=release -Daccel=omp_gpu -Dgpu_arch=<cc60|cc80> \
    -Daccel_hip=true -Daccel_blas=cublas -Dcpu_baseline=x86-64-v3 -Dkernel_precision=single
FC=nvfortran CC=nvc meson setup b64 --buildtype=release -Daccel=omp_gpu -Dgpu_arch=<cc60|cc80> \
    -Daccel_hip=true -Daccel_blas=cublas -Dcpu_baseline=x86-64-v3
ninja -C b32 benchmark_csp_batch benchmark_ssp_operator
ninja -C b64 benchmark_csp_batch benchmark_ssp_operator
```

Sanity check before sweeping (must pass; report if not): `OMP_TARGET_OFFLOAD=MANDATORY meson test -C b32 --suite unit`.

## 2. Record the hardware (put the output in the report)

```sh
nvidia-smi -q | head -60                      # model, driver, clocks, memory, PCIe gen/width
nvidia-smi --query-gpu=name,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.current,clocks.max.sm,clocks.max.mem --format=csv
lscpu | head -25; free -g; nvcc --version | tail -2; nvfortran --version | head -2
```

## 3. The sweeps (each: `tools/tuning/sweep.py`, 5 repeats, results saved as JSON)

Set `E="--env OMP_TARGET_OFFLOAD=MANDATORY"` and `R=results-<device>` (create the directory).

**3a. Noise floor** (two identical series: the difference between them is noise):

```sh
tools/tuning/sweep.py $E --config full --sizes 16,64,256,1024 --repeats 5 \
    --series a=b32/benchmark_csp_batch --series b=b32/benchmark_csp_batch --out $R/noise-fp32.json
```

**3b. `HOST_SFH_MAX`** (re-measures M1 of `doc/TUNING.md`: below the crossover the SFH weights are
formed on the host): `0` puts every batch on the device, `1000000` every batch on the host.

```sh
for p in b32 b64; do
tools/tuning/sweep.py $E --config full --sizes 8,16,32,64,128,256,512,1024 --repeats 5 \
    --series device=$p/benchmark_csp_batch@OPENSPS_TUNE_HOST_SFH_MAX=0 \
    --series host=$p/benchmark_csp_batch@OPENSPS_TUNE_HOST_SFH_MAX=1000000 --out $R/host_sfh_max-$p.json
done
```

**3c. `SSP_BATCH_CHUNK`** (models per chunk of the batched SSP; the IMF-per-model configs use it),
one batch size, FP32 and FP64:

```sh
for p in b32 b64; do
tools/tuning/sweep.py $E --config full_imf --sizes 10000 --repeats 5 --metrics 'batch_10000_mags_us_per_model' \
    --series c2048=$p/benchmark_csp_batch \
    --series c256=$p/benchmark_csp_batch@OPENSPS_TUNE_SSP_BATCH_CHUNK=256 \
    --series c512=$p/benchmark_csp_batch@OPENSPS_TUNE_SSP_BATCH_CHUNK=512 \
    --series c1024=$p/benchmark_csp_batch@OPENSPS_TUNE_SSP_BATCH_CHUNK=1024 \
    --series c4096=$p/benchmark_csp_batch@OPENSPS_TUNE_SSP_BATCH_CHUNK=4096 \
    --series c8192=$p/benchmark_csp_batch@OPENSPS_TUNE_SSP_BATCH_CHUNK=8192 \
    --series c16384=$p/benchmark_csp_batch@OPENSPS_TUNE_SSP_BATCH_CHUNK=16384 --out $R/ssp_batch_chunk-$p.json
done
```

**3d. `PINNED_CHUNK_BYTES`** (re-measures M2): read `doc/TUNING.md` M2 and `doc/GPU_RESULTS.md`,
"Spectra copy-back", and measure the same way (the pinned copy applies to the spectra metrics,
`batch_N_spec_us_per_model`, in a build that uses the pinned path). If you cannot tell from the
code how to enable the pinned path, say so and skip this sweep: do not guess.

**3e. Baseline table** (us per model; this is what formula defaults will later be fitted to):

```sh
for c in full full_smooth full_z full_imf full_z_imf; do
tools/tuning/sweep.py $E --config $c --sizes 1,10,100,1000,10000 --repeats 3 \
    --metrics 'batch_\d+_(mags|spec)_us_per_model' --series fp32=b32/benchmark_csp_batch \
    --series fp64=b64/benchmark_csp_batch --out $R/baseline-$c.json
done
```

**3f. Host-to-device bandwidth** (PCIe): if `bandwidthTest` (CUDA samples) is available run
`bandwidthTest --memory=pinned --mode=shmoo` and `--memory=pageable`; save the output to `$R/`.
Otherwise skip it and say so.

## 4. Report

Commit nothing in the source tree. Put the JSON files, the hardware record of step 2 and the console
output in a directory `tools/tuning/results/<device>-<YYYY-MM-DD>/`, commit it on a new branch
`study/<device>-data` and push that branch. Then write the report (what you ran, what failed, what you
skipped and why) with, per sweep: the best value, the spread against the default, and whether the
difference exceeds the noise measured in 3a. State plainly what you did not measure. If a command does
not work as written, fix your invocation, say what you changed, and do not change the tools.
