project('fsps', ['fortran', 'c'],
  version : '3.2.0',
  license : 'MIT',
  license_files : ['LICENSE.md'],
  default_options : [
    'buildtype=release', 
    'warning_level=0',
    'default_library=shared'
  ],
  meson_version : '>=1.1',
)

# ===================================
# Compiler Configuration
# ===================================

fc = meson.get_compiler('fortran')
fc_id = fc.get_id()
host_cpu = host_machine.cpu_family()

# Universal arguments
add_project_arguments('-cpp', language : 'fortran')

# GFortran-specific arguments
if fc_id == 'gcc'
    add_project_arguments('-ffree-line-length-132', language : 'fortran')
endif

# Installation paths
prefix = get_option('prefix')
libdir = get_option('libdir')
includedir = get_option('includedir')
datadir = get_option('datadir')

# -----------------------------------
# Hardware Acceleration (OpenACC)
# -----------------------------------

accel_target = get_option('accel')
accel_args = []
accel_defs = []
if accel_target == 'omp_gpu' and fc_id in ['llvm-flang', 'flang']
    # LLVM flang (AMD and other GPUs): -Dgpu_arch is an offload target such as gfx1103 or gfx942;
    # system-specific device-library paths go in -Dfortran_args / -Dfortran_link_args
    # (doc/GPU_PLAN.md, "AMD GPUs with LLVM flang")
    accel_args = ['-fopenmp', '--offload-arch=' + get_option('gpu_arch')]
    accel_defs = ['-DOMP_TARGET']
elif accel_target != 'none'
    if fc_id != 'nvidia_hpc'
        warning('Hardware acceleration requested but compiler does not support OpenACC. Ignoring accel option.')
    else
        if accel_target == 'multicore'
            accel_args = ['-acc=multicore']
            accel_defs = ['-DACC_MULTICORE']
        elif accel_target == 'omp_gpu'
            # OpenMP target offload of the SSP template product and the whole batched CSP
            # (compute_csp_batch, batch photometry and indices); OpenACC is off, so the rest
            # (single CSP calls, the other SSP stages) runs on the host. Run with
            # OMP_TARGET_OFFLOAD=MANDATORY to catch a silent host fallback.
            accel_args = ['-mp=gpu', '-gpu=' + get_option('gpu_arch')]
            accel_defs = ['-DOMP_TARGET', '-DFSPS_PINNED']
        elif accel_target == 'gpu'
            # Fat binary targeting Volta, Ampere, Hopper, plus PTX fallback
            accel_args = ['-acc=gpu', '-gpu=' + get_option('gpu_arch')]
            accel_defs = ['-DACC_GPU', '-DFSPS_PINNED']
            if get_option('cublas')
                accel_args += ['-cudalib=cublas']
                accel_defs += ['-DFSPS_CUBLAS']
            endif
        endif
    endif
endif

# NVIDIA GPU builds: page-locked staging for the batch's spectra copy-back (src/core/fsps_pinned.c,
# the CUDA driver resolved at run time)
pinned_sources = []
dl_dep = []
if accel_defs.contains('-DFSPS_PINNED')
    pinned_sources = files('src/core/fsps_pinned.c')
    dl_dep = meson.get_compiler('c').find_library('dl', required : false)
endif

# Create a dependency object that explicitly propagates these flags
# to both compile and link stages across all consuming targets.
accel_dep = declare_dependency(
    compile_args : accel_args + accel_defs,
    link_args : accel_args
)

# -----------------------------------
# CPU Parallelism (OpenMP)
# -----------------------------------

# Native threads: the setup-cache lock (src/core/fsps_setup_lock.c) is a pthread mutex
threads_dep = dependency('threads')

omp_dep = []
if get_option('enable_openmp')
    # Let Meson natively find OpenMP for both Fortran and C
    omp_dep = dependency('openmp', required: true)
endif

# -----------------------------------
# CPU Architecture Optimization
# -----------------------------------
# Kernel precision: FP32 per-pixel kernels (see src/core/fsps_precision.f90)
kernel_precision = get_option('kernel_precision')
if kernel_precision == 'single'
    add_project_arguments('-DKERNEL_SINGLE_PRECISION', language: 'fortran')
endif

cpu_baseline = get_option('cpu_baseline')
if get_option('buildtype') == 'release' and fc_id == 'gcc'
    if cpu_baseline == 'native'
        add_project_arguments('-march=native', language: 'fortran')
    elif cpu_baseline == 'x86-64-v3' and host_cpu == 'x86_64'
        add_project_arguments('-march=x86-64-v3', language: 'fortran')
    endif
endif

# ===================================
# Source Definitions
# ===================================

lib_sources = files(
    # Language Binding Modules
    'src/bindings/c/fsps_c_api.f90',

    # Core FSPS Modules
    'src/core/fsps_cache.f90',
    'src/core/fsps_setup_lock.c',
    'src/core/fsps_constants.f90',
    'src/core/fsps_context_types.f90',
    'src/core/fsps_context.f90',
    'src/core/fsps_environment.f90',
    'src/core/fsps_initialization.f90',
    'src/core/fsps_strings.f90',
    'src/core/fsps_api.f90',
    'src/core/fsps_precision.f90',
    'src/core/fsps_types.f90',

    # IO Module
    'src/io/fsps_io.f90',
    'src/io/fsps_json.f90',
    'src/io/fsps_datalib.f90',

    # Math Modules
    'src/math/fsps_integration.f90',
    'src/math/fsps_interpolation.f90',
    'src/math/fsps_special_functions.f90',

    # Physics Modules
    'src/physics/fsps_cosmology.f90',
    'src/physics/fsps_csp.f90',
    'src/physics/fsps_csp_batch.f90',
    'src/physics/fsps_scenario_batch.f90',
    'src/physics/fsps_dust.f90',
    'src/physics/fsps_gas.f90',
    'src/physics/fsps_imf.f90',
    'src/physics/fsps_sfh.f90',
    'src/physics/fsps_ssp.f90',
    'src/physics/fsps_ssp_operator.f90',
    'src/physics/fsps_stellar_modifications.f90',

    # Spectra Modules
    'src/spectra/fsps_batch_photometry.f90',
    'src/spectra/fsps_photometry.f90',
    'src/spectra/fsps_smoothing.f90',
    'src/spectra/fsps_spectral_indices.f90',
    'src/spectra/fsps_spectral_library.f90',
)

# ===================================
# Main Library Target
# ===================================

# This builds libfsps.so (and handles .mod generation automatically)
libfsps = library('fsps',
    lib_sources + pinned_sources,
    version : meson.project_version(),
    soversion : '3',
    install : true,
    dependencies : [omp_dep, accel_dep, threads_dep, dl_dep],
    # The library's C sources (setup-cache lock, pinning); link with the Fortran compiler so its
    # runtime (libgfortran, flang runtime, ...) is always included
    link_language : 'fortran',
)

# Declare a dependency object for executables/tests to use
# This ensures they find the generated .mod files and link to the lib
fsps_dep = declare_dependency(
    link_with : libfsps,
    include_directories : include_directories('include'),
    dependencies : [omp_dep, accel_dep, threads_dep]
)
# dependency('fsps') in a project that has this one as a subproject
meson.override_dependency('fsps', fsps_dep)

# ===================================
# Tests
# ===================================

# Tests are optional (-Dtests=false), e.g. when the library is built as a subproject
if get_option('tests')
# (the test section below is not indented, to keep it diffable)

# 1. Test Utilities (Helper for other tests)
test_utils_lib = static_library(
    'test_utils', 
    'tests/test_utils.f90', 
    dependencies : fsps_dep
)

test_utils_dep = declare_dependency(link_with : test_utils_lib)

# The data root of the regression and stress suites: -Ddata_home=<root> (a legacy tree containing data/, or a
# standard-format root such as one unpacked by tools/fetch_data.py); default: this source tree.
# If the option is empty, OPENSPS_DATA_HOME as set when configuring, else the source tree.
data_home = get_option('data_home')
if data_home == ''
    data_home = run_command('sh', '-c', 'printf %s "$OPENSPS_DATA_HOME"', check : false).stdout().strip()
endif
if data_home == ''
    data_home = meson.project_source_root()
endif
message('Data root of the regression and stress suites: ' + data_home)

# 2. Unit Tests
unit_test_sources = files(
    'tests/core/test_fsps_cache.f90',
    'tests/core/test_fsps_context.f90',
    'tests/core/test_fsps_context_types.f90',
    'tests/core/test_fsps_strings.f90',
    'tests/core/test_fsps_types.f90',

    'tests/io/test_fsps_io.f90',

    'tests/math/test_fsps_integration.f90',
    'tests/math/test_fsps_interpolation.f90',
    'tests/math/test_fsps_special_functions.f90',

    'tests/physics/test_fsps_cosmology.f90',
    'tests/physics/test_fsps_csp.f90',
    'tests/physics/test_fsps_dust.f90',
    'tests/physics/test_fsps_gas.f90',
    'tests/physics/test_fsps_imf.f90',
    'tests/physics/test_fsps_sfh.f90',
    'tests/physics/test_fsps_ssp.f90',
    'tests/physics/test_fsps_stellar_modifications.f90',

    'tests/spectra/test_fsps_photometry.f90',
    'tests/spectra/test_fsps_smoothing.f90',
    'tests/spectra/test_fsps_spectral_indices.f90',
    'tests/spectra/test_fsps_spectral_library.f90',

    'tests/test_fsps.f90' # Main driver
)

tests_units_exe = executable(
    'tests_units', 
    unit_test_sources, 
    dependencies : [fsps_dep, test_utils_dep],
    link_language : 'fortran'
)
# One TAP test per module: meson runs them in parallel and reports every assertion
unit_env = environment()
unit_env.set('FSPS_TEST_TAP', '1')
# Unit tests that read real data use the small standard-format test library, never data/
unit_env.set('FSPS_TEST_LIBRARY', join_paths(meson.project_source_root(), 'tests/data/fsps-test-data'))
unit_modules = [
    'fsps_cache', 'fsps_context', 'fsps_context_types', 'fsps_strings', 'fsps_types', 'fsps_io',
    'fsps_integration', 'fsps_interpolation', 'fsps_special_functions', 'fsps_cosmology', 'fsps_csp',
    'fsps_dust', 'fsps_gas', 'fsps_imf', 'fsps_sfh', 'fsps_ssp', 'fsps_stellar_modifications',
    'fsps_photometry', 'fsps_smoothing', 'fsps_spectral_indices', 'fsps_spectral_library',
]
foreach unit_module : unit_modules
    test('unit_' + unit_module.substring(5), tests_units_exe,
        args : [unit_module],
        env : unit_env,
        protocol : 'tap',
        suite : 'unit'
    )
endforeach

# 3. Other Test Drivers
# Fortran programs link with the Fortran compiler: meson otherwise picks the C linker because
# libfsps has a C source, and nvc (HPC SDK 25.1) does not pull in the Fortran runtime
test_runner_exe = executable('test_runner', 'tests/test_runner.f90', dependencies : fsps_dep, link_language : 'fortran')
executable('test_cache', 'tests/test_cache.f90', dependencies : fsps_dep, link_language : 'fortran')
executable('generate_test_data', 'tests/generate_test_data.f90', dependencies : fsps_dep, link_language : 'fortran')
executable('generate_opensps_fixtures', 'tests/generate_opensps_fixtures.f90', dependencies : fsps_dep, link_language : 'fortran')
executable('benchmark_hotpaths', 'tests/benchmark_hotpaths.f90', dependencies : fsps_dep, link_language : 'fortran')
executable('benchmark_phases', 'tests/benchmark_phases.f90', dependencies : fsps_dep, link_language : 'fortran')
executable('benchmark_csp_batch', 'tests/benchmark_csp_batch.f90', dependencies : fsps_dep, link_language : 'fortran')
executable('benchmark_ssp_operator', 'tests/benchmark_ssp_operator.f90', dependencies : fsps_dep, link_language : 'fortran')

if get_option('enable_stress_testing')
    stress_test_exe = executable('stress_test', 'tests/stress_test.f90', dependencies : fsps_dep, link_language : 'fortran')
    stress_env = environment()
    stress_env.set('OPENSPS_DATA_HOME', data_home)
    stress_env.set('FSPS_DATA_HOME', data_home)
    stress_env.set('SPS_HOME', data_home)
    test('stress_test', stress_test_exe,
        suite : 'stress',
        env : stress_env,
        # Phase D serializes repeated full setups (~160 s each under ThreadSanitizer)
        timeout : 2400
    )
endif

# 3a. Regression Tests (run through Meson test framework)
fs = import('fs')
# The C3K library (~600 MB) may be absent (e.g. a sparse CI clone, or a release fetched without the
# optional archive); its tests are then skipped rather than failing on missing files.
c3k_available = (fs.is_dir(join_paths(data_home, 'data/spectra/C3K')) or
                 fs.is_dir(join_paths(data_home, 'libraries/stellar_spectra/c3k_afe+0.0')))
if not c3k_available
    message('C3K not found in ' + data_home + ': C3K regression tests are skipped')
endif
regression_env = environment()
if kernel_precision == 'single'
    # FP32 kernels against the FP64 references: spectra and magnitudes agree to ~1e-6, weak
    # spectral indices (values near zero) to ~2e-5 in absolute terms
    regression_env.set('FSPS_TEST_RTOL', '1.0E-4')
    regression_env.set('FSPS_TEST_INDX_ATOL', '1.0E-4')
else
    regression_env.set('FSPS_TEST_RTOL', '1.0E-5')
endif
# All three variables: OPENSPS_DATA_HOME or SPS_HOME, if exported by the user, would otherwise take
# precedence
regression_env.set('OPENSPS_DATA_HOME', data_home)
regression_env.set('FSPS_DATA_HOME', data_home)
regression_env.set('SPS_HOME', data_home)
# Programs built on tests/test_utils.f90 report each assertion as TAP
regression_env.set('FSPS_TEST_TAP', '1')

# Integration test: contexts sharing a setup cache must not leak state into each other
context_isolation_exe = executable('test_context_isolation', 'tests/test_context_isolation.f90',
    dependencies : [fsps_dep, test_utils_dep],
    link_language : 'fortran'
)
test('context_isolation', context_isolation_exe,
    protocol : 'tap',
    env : regression_env,
    suite : 'regression',
    timeout : 600
)

# Integration test: outputs checked against independent physical identities and published values
physics_oracles_exe = executable('test_physics_oracles', 'tests/test_physics_oracles.f90',
    dependencies : [fsps_dep, test_utils_dep],
    link_language : 'fortran'
)
test('physics_oracles', physics_oracles_exe,
    protocol : 'tap',
    env : regression_env,
    suite : 'regression',
    timeout : 900
)

# Integration test: the batched CSP (fsps_csp_batch) against compute_csp_scenario, model by model
csp_batch_exe = executable('test_csp_batch', 'tests/test_csp_batch.f90',
    dependencies : [fsps_dep, test_utils_dep],
    link_language : 'fortran'
)
test('csp_batch', csp_batch_exe,
    protocol : 'tap',
    env : regression_env,
    suite : 'regression',
    timeout : 900
)

# Integration test: the batched SSP operator (fsps_ssp_operator) against generate_ssp_grid, for
# many IMFs from one IMF-independent operator
ssp_operator_exe = executable('test_ssp_operator', 'tests/test_ssp_operator.f90',
    dependencies : [fsps_dep, test_utils_dep],
    link_language : 'fortran'
)
test('ssp_operator', ssp_operator_exe,
    protocol : 'tap',
    env : regression_env,
    suite : 'regression',
    timeout : 900
)

# Regression references (doc/DATA_FORMAT.md, Section 7): the data release's libraries
# references/sps and references/param in the data root, else the in-tree copies in tests/data while
# they exist. Without either the tests that use them are skipped (exit 77 / TAP SKIP), not failed;
# tests/data/references.json pins the expected versions (test references_pin).
sps_ref_dir = join_paths(data_home, 'libraries/references/sps')
if not fs.is_file(join_paths(sps_ref_dir, 'manifest.json'))
    sps_ref_dir = join_paths(meson.project_source_root(), 'tests/data')
    if not fs.is_file(join_paths(sps_ref_dir, 'sps_ref_MILES-1_MIST-1.bin'))
        sps_ref_dir = join_paths(data_home, 'libraries/references/sps')
        message('Regression references references/sps not found in ' + data_home + ' or tests/data: those tests are skipped')
    endif
endif
param_ref_dir = join_paths(data_home, 'libraries/references/param')
if not fs.is_file(join_paths(param_ref_dir, 'manifest.json'))
    param_ref_dir = join_paths(meson.project_source_root(), 'tests/data/param_ref')
    if not fs.is_dir(param_ref_dir)
        param_ref_dir = join_paths(data_home, 'libraries/references/param')
        message('Regression references references/param not found in ' + data_home + ' or tests/data: those tests are skipped')
    endif
endif
message('Regression references: ' + sps_ref_dir + ', ' + param_ref_dir)
python3_ref = find_program('python3', required : false)
if python3_ref.found()
    test('references_pin', python3_ref,
        args : [join_paths(meson.project_source_root(), 'tests/check_references.py'),
                join_paths(meson.project_source_root(), 'tests/data/references.json'), sps_ref_dir, param_ref_dir],
        suite : 'regression',
        timeout : 120
    )
endif

# Parameterized regression over the parity case matrix (shared with generate_opensps_fixtures.sh),
# sharded by case family so that meson runs the shards in parallel
param_regression_exe = executable('test_param_regression', 'tests/test_param_regression.f90',
    dependencies : [fsps_dep, test_utils_dep],
    link_language : 'fortran'
)
param_filter_args = c3k_available ? [] : ['--skip-spec', 'c3k']
foreach shard : ['ssp_', 'csp_', 'lib_']
    test('param_regression_' + shard.strip('_'), param_regression_exe,
        protocol : 'tap',
        args : [
            join_paths(meson.project_source_root(), 'tests/data/param_cases.psv'),
            param_ref_dir,
            '--only', shard
        ] + param_filter_args,
        env : regression_env,
        suite : 'regression',
        timeout : 1200
    )
endforeach

test('regression_miles_mist',
        test_runner_exe,
        args : [
            join_paths(sps_ref_dir, 'sps_ref_MILES-1_MIST-1.bin'),
            '--isoc', 'mist', '--spec', 'miles'
        ],
        env : regression_env,
        suite : 'regression',
        timeout : 300
)

test('regression_basel_padova',
        test_runner_exe,
        args : [
            join_paths(sps_ref_dir, 'sps_ref_MILES-0_MIST-0_BASEL-1_PADOVA-1.bin'),
            '--isoc', 'pdva', '--spec', 'basel'
        ],
        env : regression_env,
        suite : 'regression',
        timeout : 300
)

test('regression_themis_dust',
        test_runner_exe,
        args : [
            join_paths(sps_ref_dir, 'sps_ref_MILES-1_MIST-1_THEMIS-1_DL07-0.bin'),
            '--isoc', 'mist', '--spec', 'miles', '--dust', 'themis'
        ],
        env : regression_env,
        suite : 'regression',
        timeout : 300
)

if c3k_available
    test('regression_c3k_mist',
            test_runner_exe,
            args : [
                join_paths(sps_ref_dir, 'sps_ref_MILES-0_C3K-1_MIST-1.bin'),
                '--isoc', 'mist', '--spec', 'c3k_afe+0.0'
            ],
            env : regression_env,
            suite : 'regression',
            timeout : 300
    )
endif

test('regression_bpass',
        test_runner_exe,
        args : [
            join_paths(sps_ref_dir, 'sps_ref_MIST-0_BPASS-1.bin'),
            '--isoc', 'bpss'
        ],
        env : regression_env,
        suite : 'regression',
        timeout : 300
)

# Bit-for-bit parity of the legacy layout and the standard data format (doc/DATA_FORMAT.md).
# Needs a converted data root: -Dstandard_data_home=<root> (tools/fsps_data/convert_legacy.py);
# skipped when the root has no opensps-data.json (or fsps-data.json).
data_parity_exe = executable('test_data_parity', 'tests/test_data_parity.f90',
    dependencies : [fsps_dep, test_utils_dep],
    link_language : 'fortran'
)
standard_data_home = get_option('standard_data_home')
if standard_data_home == ''
    standard_data_home = join_paths(meson.project_source_root(), 'opensps-data')
endif
# The legacy side is the source tree's data/, so the suite exists only while the tree has it
if fs.is_dir('data')
test('data_parity', data_parity_exe,
    protocol : 'tap',
    args : [meson.project_source_root(), standard_data_home] + (c3k_available ? [] : ['--skip-spec', 'c3k_afe+0.0']),
    env : regression_env,
    suite : 'data_parity',
    timeout : 1800
)
endif

# Setup and synthesis from the small standard-format test library committed in the repository
# (tests/data/fsps-test-data, no other data needed), and [alpha/Fe] axis selection on variants
# derived at test time (tools/fsps_data/make_afe_variants.py)
standard_library_exe = executable('test_standard_library', 'tests/test_standard_library.f90',
    dependencies : [fsps_dep, test_utils_dep],
    link_language : 'fortran'
)
python3 = find_program('python3', required : false)
if python3.found()
    test('unit_standard_library', python3,
        args : [join_paths(meson.project_source_root(), 'tests/run_standard_library_test.py'),
                standard_library_exe, join_paths(meson.project_source_root(), 'tests/data/fsps-test-data'),
                meson.current_build_dir()],
        env : unit_env,
        protocol : 'tap',
        suite : 'unit',
        timeout : 300
    )
    # The Python data tools (format round trip, converters of the upstream [alpha/Fe] layouts)
    test('unit_data_tools', python3,
        args : [join_paths(meson.project_source_root(), 'tools/fsps_data/test_fsps_data.py')],
        suite : 'unit',
        timeout : 120
    )
endif

# 4. C Tests
executable('test_c_driver', 'tests/test_c_driver.c', 
    dependencies : fsps_dep,
    include_directories : include_directories('include')
)
test_c_batch_exe = executable('test_c_batch', 'tests/test_c_batch.c',
    dependencies : [fsps_dep, meson.get_compiler('c').find_library('m', required : false)],
    include_directories : include_directories('include')
)
test('c_batch', test_c_batch_exe,
    env : regression_env,
    suite : 'regression',
    timeout : 600
)
test_c_batch_parts_exe = executable('test_c_batch_parts', 'tests/test_c_batch_parts.c',
    dependencies : [fsps_dep, meson.get_compiler('c').find_library('m', required : false)],
    include_directories : include_directories('include')
)
test('c_batch_parts', test_c_batch_parts_exe,
    env : regression_env,
    suite : 'regression',
    timeout : 600
)
test_c_csp_exe = executable('test_c_csp', 'tests/test_c_csp.c',
    dependencies : [fsps_dep, meson.get_compiler('c').find_library('m', required : false)],
    include_directories : include_directories('include')
)
test('c_csp', test_c_csp_exe,
    env : regression_env,
    suite : 'regression',
    timeout : 600
)

endif # tests

# ===================================
# Installation (PkgConfig & Data)
# ===================================

# Install Headers
install_headers('include/fsps.h')

# Install Data (recursively copies the directory; 1.3 GB, off with -Dinstall_data=false, e.g. for
# a Python wheel that builds the library as a subproject; skipped when the tree has no data/)
if get_option('install_data') and fs.is_dir('data')
    install_subdir('data', install_dir : join_paths(datadir, 'fsps'))
endif

# Pkg-config generation
pkg = import('pkgconfig')
pkg.generate(libfsps,
    name : 'fsps',
    description : 'Flexible Stellar Population Synthesis',
    url : 'https://github.com/cconroy20/fsps'
)
