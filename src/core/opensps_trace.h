#ifndef OPENSPS_TRACE_H
#define OPENSPS_TRACE_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Platform-neutral Range Tracing (ROCTX on AMD, NVTX on NVIDIA, no-op otherwise).
 * When not compiled with OPENSPS_ENABLE_TRACE, these inline to zero-overhead no-ops.
 */

#if defined(OPENSPS_ENABLE_TRACE)

#if defined(__HIP_PLATFORM_AMD__) || defined(OPENSPS_ROCTX)
#include <rocprofiler-sdk-roctx/roctx.h>
static inline void opensps_trace_range_push(const char *name) {
    roctxRangePush(name);
}
static inline void opensps_trace_range_pop(void) {
    roctxRangePop();
}
#elif defined(__NVCC__) || defined(OPENSPS_NVTX)
#include <nvtx3/nvToolsExt.h>
static inline void opensps_trace_range_push(const char *name) {
    nvtxRangePushA(name);
}
static inline void opensps_trace_range_pop(void) {
    nvtxRangePop();
}
#else
static inline void opensps_trace_range_push(const char *name) { (void)name; }
static inline void opensps_trace_range_pop(void) {}
#endif

#else /* !defined(OPENSPS_ENABLE_TRACE) */

static inline void opensps_trace_range_push(const char *name) { (void)name; }
static inline void opensps_trace_range_pop(void) {}

#endif

#ifdef __cplusplus
}
#endif

#endif /* OPENSPS_TRACE_H */
