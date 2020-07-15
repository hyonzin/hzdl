#include "util.h"

int _get_num_element(layer* p) {
    return (p->c * p->h * p->w);
}

void _safe_free(float** ptr) {
    if (ptr != NULL && *ptr != NULL) {
        free(*ptr);
        *ptr = NULL;
    }
}

float _random_float() {
    return (float)(rand() % 10000 - 5000) / 100000;
}
