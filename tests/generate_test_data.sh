#!/bin/bash

# Generate FSPS test reference data for multiple library configurations.
# This script must be run from the tests/ directory.
# Usage: ./generate_test_data.sh

set -e

# ============================================================================
# 1. Resolve Paths
# ============================================================================

# Get the absolute directory where this script is located (tests/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Resolve the project root (one level up from tests/)
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

# Define critical paths relative to the resolved root
TEST_DATA_DIR="$SCRIPT_DIR/data"
BUILD_DIR="$PROJECT_ROOT/build"
GENERATOR="$BUILD_DIR/generate_test_data"

# Ensure FSPS finds data files by setting the data root to the project root (OPENSPS_DATA_HOME is
# searched first, then SPS_HOME, then FSPS_DATA_HOME)
export OPENSPS_DATA_HOME="$PROJECT_ROOT"
export FSPS_DATA_HOME="$PROJECT_ROOT"

# ============================================================================
# 2. Configuration
# ============================================================================

# Format: "Legacy_File_Suffix|Runtime_Args"
declare -a configurations=(
    "MILES-1_MIST-1|--isoc mist --spec miles"
    "MILES-0_MIST-0_BASEL-1_PADOVA-1|--isoc pdva --spec basel"
    "MILES-1_MIST-1_THEMIS-1_DL07-0|--isoc mist --spec miles --dust themis"
    "MILES-0_C3K-1_MIST-1|--isoc mist --spec c3k_afe+0.0"
    "MIST-0_BPASS-1|--isoc bpss --spec bpass"
)

# Create the data output directory if it doesn't exist
mkdir -p "$TEST_DATA_DIR"

# ============================================================================
# 3. Compilation
# ============================================================================

echo "=========================================================="
echo "Compiling generator with Meson..."
echo "Project Root: $PROJECT_ROOT"
echo "=========================================================="

# Switch to the project root context to run build commands
pushd "$PROJECT_ROOT" > /dev/null

# Setup build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo "Build directory not found. Running meson setup..."
    meson setup build
fi

# Compile the specific target
meson compile -C build generate_test_data

if [ ! -f "$GENERATOR" ]; then
    echo "ERROR: Compilation failed or executable not found at $GENERATOR"
    popd > /dev/null
    exit 1
fi

# ============================================================================
# 4. Generation Loop
# ============================================================================

for config in "${configurations[@]}"; do
    IFS="|" read -r suffix args <<< "$config"
    
    echo "=========================================================="
    echo "Running configuration: $args"
    echo "Output: tests/data/sps_ref_${suffix}.bin"
    echo "=========================================================="

    # Run the generator
    # The binary will write 'sps_test_output.bin' to the CURRENT directory (Project Root)
    $GENERATOR $args
    
    # Move and rename output to the resolved tests/data folder
    if [ -f "sps_test_output.bin" ]; then
        mv sps_test_output.bin "$TEST_DATA_DIR/sps_ref_${suffix}.bin"
        echo "Created: $TEST_DATA_DIR/sps_ref_${suffix}.bin"
    else
        echo "ERROR: Output file not generated for $args"
        # Cleanup context before exiting
        popd > /dev/null
        exit 1
    fi
    
    echo ""
done

# Restore original directory context
popd > /dev/null

echo "Done."
