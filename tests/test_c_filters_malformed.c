/* Installs the filter library named by argv[1] in a set-up context and prints the status. A malformed library
 * stops the program with a message on stderr (the readers' error stop); tests/run_filters_malformed_test.py runs
 * this once per kind of malformation, and checks the exit status and the message. */
#include "opensps.h"

#include <stdio.h>

int main(int argc, char **argv) {
    int h = 0, status = 0;
    if (argc < 2) { printf("usage: test_c_filters_malformed LIBRARY\n"); return 2; }
    opensps_context_create(&h, &status);
    opensps_context_setup(-1, "mist", "miles", "DL07", h, &status);
    if (status != 0) { printf("setup failed (%d)\n", status); return 3; }
    opensps_context_set_filters_library(h, argv[1], 1, 0, 0, &status);
    printf("status %d\n", status);
    opensps_context_destroy(h, &status);
    return 0;
}
