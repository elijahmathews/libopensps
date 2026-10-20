/*
 * Page-locking (pinning) of host buffers for faster device-to-host copies in CUDA builds.
 *
 * The batched CSP copies its spectra back from the device; into pageable memory the CUDA driver
 * stages the copy through its own pinned buffers (~2.4 GB/s on the A100 node), into pinned memory
 * the DMA runs at the PCIe rate. The driver API is resolved at run time (dlopen of libcuda), so
 * the library does not link against CUDA and the calls quietly do nothing where the driver is
 * missing; the OpenACC runtime has already made its context current on the calling thread.
 * Built only for -Daccel=gpu (nvfortran OpenACC / OpenMP target on NVIDIA GPUs).
 */

#include <dlfcn.h>
#include <pthread.h>
#include <stddef.h>
#include <stdlib.h>

typedef int (*register_fn)(void *, size_t, unsigned int);
typedef int (*unregister_fn)(void *);
typedef int (*d2h_async_fn)(void *, unsigned long long, size_t, void *);

static register_fn fsps_cu_register;
static unregister_fn fsps_cu_unregister;
static d2h_async_fn fsps_cu_d2h_async;
static pthread_once_t fsps_pinned_once = PTHREAD_ONCE_INIT;

static void fsps_pinned_init(void)
{
    /* FSPS_NO_PIN=1 keeps the buffers pageable (for comparisons) */
    const char *off = getenv("FSPS_NO_PIN");
    if (off && off[0] && off[0] != '0') return;
    void *lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!lib) lib = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!lib) return;
    fsps_cu_register = (register_fn)dlsym(lib, "cuMemHostRegister_v2");
    fsps_cu_unregister = (unregister_fn)dlsym(lib, "cuMemHostUnregister");
    fsps_cu_d2h_async = (d2h_async_fn)dlsym(lib, "cuMemcpyDtoHAsync_v2");
    if (!fsps_cu_register || !fsps_cu_unregister || !fsps_cu_d2h_async) {
        fsps_cu_register = NULL;
        fsps_cu_unregister = NULL;
        fsps_cu_d2h_async = NULL;
    }
}

/* Pin bytes bytes at ptr (CU_MEMHOSTREGISTER_PORTABLE). Returns 0 on success. */
int fsps_host_register(void *ptr, size_t bytes)
{
    pthread_once(&fsps_pinned_once, fsps_pinned_init);
    if (!fsps_cu_register || !ptr || bytes == 0) return -1;
    return fsps_cu_register(ptr, bytes, 0x01) == 0 ? 0 : -1;
}

/* Unpin a buffer pinned by fsps_host_register. Returns 0 on success. */
int fsps_host_unregister(void *ptr)
{
    pthread_once(&fsps_pinned_once, fsps_pinned_init);
    if (!fsps_cu_unregister || !ptr) return -1;
    return fsps_cu_unregister(ptr) == 0 ? 0 : -1;
}

/* Device-to-host copy of bytes bytes from device address src to host address dst (pinned by
 * fsps_host_register) on CUDA stream stream (an OpenACC queue's, acc_get_cuda_stream), so it
 * runs as one DMA into the buffer instead of through the OpenACC runtime's staging buffers
 * (which it uses even for registered memory: ~4.6 instead of ~11 GB/s on the A100 node).
 * Returns 0 on success; the caller falls back to an OpenACC update otherwise. */
int fsps_memcpy_d2h_async(void *dst, const void *src, size_t bytes, void *stream)
{
    pthread_once(&fsps_pinned_once, fsps_pinned_init);
    if (!fsps_cu_d2h_async || !dst || !src) return -1;
    return fsps_cu_d2h_async(dst, (unsigned long long)(size_t)src, bytes, stream) == 0 ? 0 : -1;
}
