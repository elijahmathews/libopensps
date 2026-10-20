/*
 * Process-wide lock serializing access to the shared setup cache (fsps_cache).
 *
 * The setup cache is shared by every context in the process, so setup
 * (cache lookup, population of a new entry, reference counting) and release
 * must be serialized for all threads, whether they come from OpenMP or from
 * a host application calling the C API with its own pthreads. A pthread
 * mutex covers both cases (OpenMP runtimes are built on native threads) and
 * is understood by ThreadSanitizer.
 *
 * The mutex is recursive because fsps_initialize_data, which holds the lock,
 * releases the context's previous cache entry through
 * fsps_cache_release_setup, which takes it again.
 */

#include <pthread.h>

static pthread_mutex_t fsps_setup_mutex;
static pthread_once_t fsps_setup_mutex_once = PTHREAD_ONCE_INIT;

/* One-time initialization of the recursive mutex (thread-safe via pthread_once). */
static void fsps_setup_mutex_init(void)
{
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&fsps_setup_mutex, &attr);
    pthread_mutexattr_destroy(&attr);
}

/* Acquire the setup-cache lock (re-entrant for the owning thread). */
void fsps_setup_lock(void)
{
    pthread_once(&fsps_setup_mutex_once, fsps_setup_mutex_init);
    pthread_mutex_lock(&fsps_setup_mutex);
}

/* Release the setup-cache lock. */
void fsps_setup_unlock(void)
{
    pthread_mutex_unlock(&fsps_setup_mutex);
}
