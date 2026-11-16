/* CI debugging: the cost of the operations that differ between CPU models in an instrumented (gcov) build.
 * gcc's --coverage with -pthread updates every counter with a locked read-modify-write (-fprofile-update=prefer-atomic);
 * a locked operation costs a different number of cycles on different microarchitectures. Prints ns per operation of
 *   plain     a volatile 64-bit increment (what -fprofile-update=single compiles to, without the optimizer)
 *   locked    __atomic_fetch_add (lock xadd / lock add), uncontended
 *   cas       a compare-and-swap loop increment
 *   fp_norm   a dependent FP64 multiply-add on normal numbers
 *   fp_denorm the same on subnormal numbers (microcode assists on some CPUs)
 */
#include <stdint.h>
#include <stdio.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static volatile uint64_t vcount;
static uint64_t acount;

int main(void) {
    const long n = 200000000L;
    double t0, t1;

    t0 = now();
    for (long i = 0; i < n; i++) vcount = vcount + 1;
    t1 = now();
    printf("plain     %6.3f ns/op\n", (t1 - t0) / n * 1e9);

    t0 = now();
    for (long i = 0; i < n; i++) __atomic_fetch_add(&acount, 1, __ATOMIC_RELAXED);
    t1 = now();
    printf("locked    %6.3f ns/op\n", (t1 - t0) / n * 1e9);

    t0 = now();
    for (long i = 0; i < n / 4; i++) {
        uint64_t old = __atomic_load_n(&acount, __ATOMIC_RELAXED);
        while (!__atomic_compare_exchange_n(&acount, &old, old + 1, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
    }
    t1 = now();
    printf("cas       %6.3f ns/op\n", (t1 - t0) / (n / 4) * 1e9);

    volatile double sink;
    double s = 1.0e-3, c = 0.9999999999, d = 1.0e-9;
    t0 = now();
    for (long i = 0; i < n / 4; i++) s = s * c + d;
    t1 = now();
    sink = s;
    printf("fp_norm   %6.3f ns/op\n", (t1 - t0) / (n / 4) * 1e9);

    s = 1.0e-310; c = 0.9999999999; d = 1.0e-318;
    t0 = now();
    for (long i = 0; i < n / 4; i++) s = s * c + d;
    t1 = now();
    sink = s;
    printf("fp_denorm %6.3f ns/op (final %g)\n", (t1 - t0) / (n / 4) * 1e9, (double)sink);
    return 0;
}
