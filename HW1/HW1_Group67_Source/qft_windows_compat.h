#pragma once

/* Force-include on MSVC to build the original POSIX benchmark unchanged. */
#define _USE_MATH_DEFINES
#include <windows.h>
#include <time.h>

#define CLOCK_MONOTONIC 1

static inline int clock_gettime(int clock_id, struct timespec* ts) {
    LARGE_INTEGER count, freq;
    if (clock_id != CLOCK_MONOTONIC ||
        !QueryPerformanceCounter(&count) || !QueryPerformanceFrequency(&freq)) {
        return -1;
    }
    ts->tv_sec = (time_t)(count.QuadPart / freq.QuadPart);
    ts->tv_nsec = (long)((count.QuadPart % freq.QuadPart) * 1000000000LL / freq.QuadPart);
    return 0;
}
