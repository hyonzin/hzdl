#include "hzdl/util.h"

struct _layer;
struct timeval _tv_start, _tv_end;
int _rand_init = 0;

void _safe_free(float** ptr) {
    if (ptr != NULL && *ptr != NULL) {
        free(*ptr);
        *ptr = NULL;
    }
}

float _random_float() {
    if (_rand_init) {
        srand(time(NULL));
        _rand_init = 1;
    }
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

