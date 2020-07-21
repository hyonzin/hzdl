#pragma once

#include <stdlib.h>
#include <time.h>
#include <sys/time.h>


#define getMillisecond(start, end) \
        (end.tv_sec-start.tv_sec)*1000 + \
    (end.tv_usec-start.tv_usec)/1000.0

#define _get_num_element(p) (p->c * p->h * p->w)


void _time_start();

void _time_end();

float _get_time();

void _safe_free(float** ptr);

float _random_float();


