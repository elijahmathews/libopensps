/*
 * Page-locking (pinning) of host buffers for faster device-to-host copies in CUDA builds.
 *
 * The batched CSP copies its spectra back from the device; into pageable memory the CUDA driver
 * stages the copy through its own pinned buffers (~2.4 GB/s on the A100 node), into pinned memory
 * the DMA runs at the PCIe rate. The driver API is resolved at run time (dlopen of libcuda), so
 * the library does not link against CUDA and the calls quietly do nothing where the driver is
 * missing; the OpenMP runtime has already made its context current on the calling thread.
 * Built only for -Daccel=omp_gpu with nvfortran (NVIDIA GPUs).
 */

#include <dlfcn.h>
#include <pthread.h>
#include <stddef.h>
#include <stdlib.h>

typedef int (*register_fn)(void *, size_t, unsigned int);
typedef int (*unregister_fn)(void *);
typedef int (*d2h_async_fn)(void *, unsigned long long, size_t, void *);
typedef int (*stream_create_fn)(void **, unsigned int);
typedef int (*stream_sync_fn)(void *);

static register_fn opensps_cu_register;
static unregister_fn opensps_cu_unregister;
static d2h_async_fn opensps_cu_d2h_async;
static stream_create_fn opensps_cu_stream_create;
static stream_sync_fn opensps_cu_stream_sync;
static pthread_once_t opensps_pinned_once = PTHREAD_ONCE_INIT;

static void opensps_pinned_init(void)
{
    /* FSPS_NO_PIN=1 keeps the buffers pageable (for comparisons) */
    const char *off = getenv("FSPS_NO_PIN");
    if (off && off[0] && off[0] != '0') return;
    void *lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!lib) lib = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!lib) return;
    opensps_cu_register = (register_fn)dlsym(lib, "cuMemHostRegister_v2");
    opensps_cu_unregister = (unregister_fn)dlsym(lib, "cuMemHostUnregister");
    opensps_cu_d2h_async = (d2h_async_fn)dlsym(lib, "cuMemcpyDtoHAsync_v2");
    opensps_cu_stream_create = (stream_create_fn)dlsym(lib, "cuStreamCreate");
    opensps_cu_stream_sync = (stream_sync_fn)dlsym(lib, "cuStreamSynchronize");
    if (!opensps_cu_register || !opensps_cu_unregister || !opensps_cu_d2h_async) {
        opensps_cu_register = NULL;
        opensps_cu_unregister = NULL;
        opensps_cu_d2h_async = NULL;
    }
}

/* Pin bytes bytes at ptr (CU_MEMHOSTREGISTER_PORTABLE). Returns 0 on success. */
int opensps_host_register(void *ptr, size_t bytes)
{
    pthread_once(&opensps_pinned_once, opensps_pinned_init);
    if (!opensps_cu_register || !ptr || bytes == 0) return -1;
    return opensps_cu_register(ptr, bytes, 0x01) == 0 ? 0 : -1;
}

/* Unpin a buffer pinned by opensps_host_register. Returns 0 on success. */
int opensps_host_unregister(void *ptr)
{
    pthread_once(&opensps_pinned_once, opensps_pinned_init);
    if (!opensps_cu_unregister || !ptr) return -1;
    return opensps_cu_unregister(ptr) == 0 ? 0 : -1;
}

/* Device-to-host copy of bytes bytes from device address src to host address dst (pinned by
 * opensps_host_register) on CUDA stream stream (opensps_stream_create), as one DMA into the
 * buffer (an OpenACC runtime staged even registered memory: ~4.6 instead of ~11 GB/s on the A100
 * node). Returns 0 on success; the caller falls back to a target update otherwise. */
int opensps_memcpy_d2h_async(void *dst, const void *src, size_t bytes, void *stream)
{
    pthread_once(&opensps_pinned_once, opensps_pinned_init);
    if (!opensps_cu_d2h_async || !dst || !src) return -1;
    return opensps_cu_d2h_async(dst, (unsigned long long)(size_t)src, bytes, stream) == 0 ? 0 : -1;
}

/* A CUDA stream of the current context (CU_STREAM_NON_BLOCKING), for copies that are not ordered
 * with a runtime's own queues: the OpenMP target build's copy-back, which starts after its
 * (synchronous) kernels have finished. NULL if the driver is not available. */
void *opensps_stream_create(void)
{
    void *s = NULL;
    pthread_once(&opensps_pinned_once, opensps_pinned_init);
    if (!opensps_cu_stream_create || !opensps_cu_d2h_async) return NULL;
    return opensps_cu_stream_create(&s, 0x1) == 0 ? s : NULL;
}

/* Waits for the work on stream s (from opensps_stream_create). Returns 0 on success. */
int opensps_stream_sync(void *s)
{
    if (!opensps_cu_stream_sync || !s) return -1;
    return opensps_cu_stream_sync(s) == 0 ? 0 : -1;
}
