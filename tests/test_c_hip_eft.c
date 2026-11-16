/* The error-free transforms of the native GPU kernels survive the compiler (doc/MIXED_PRECISION.md, rule 9):
 * a rounded product fused into the sum that uses it would count its rounding error twice. The device self-test
 * (opensps_hip_selftest_eft) runs the observed-frame numerator and the fold's product-and-sum on random inputs and
 * compares them with FP64: 0 failures, or the test fails. Built only with the HIP kernels; needs a device. */
#include <stdio.h>

extern int opensps_hip_selftest_eft(int n);

int main(void) {
    int bad = opensps_hip_selftest_eft(1 << 20);
    if (bad < 0) {
        printf("c_hip_eft: the self-test could not run (no device?)\n");
        return 77;
    }
    printf("c_hip_eft: %d failures of 2^21 checks\n", bad);
    return bad == 0 ? 0 : 1;
}
