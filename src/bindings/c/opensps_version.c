#include "opensps.h"

void opensps_get_driver_version(int *major, int *minor, int *patch) {
    if (major) *major = OPENSPS_API_VERSION_MAJOR;
    if (minor) *minor = OPENSPS_API_VERSION_MINOR;
    if (patch) *patch = OPENSPS_API_VERSION_PATCH;
}

const char *opensps_get_version_string(void) {
    return OPENSPS_VERSION_STRING;
}
