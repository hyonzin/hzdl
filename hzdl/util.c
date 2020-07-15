#include "util.h"

void _safe_free(float** ptr) {
    if (ptr != NULL && *ptr != NULL) {
        free(*ptr);
        *ptr = NULL;
    }
}

float _random_float() {
    return (float)(rand() % 10000 - 5000) / 100000;
}

void _time_start() {
    gettimeofday(&_tv_start, NULL);
}

void _time_end() {
    gettimeofday(&_tv_end, NULL);
}

float _get_time() {
    return getMillisecond(_tv_start, _tv_end);
}

