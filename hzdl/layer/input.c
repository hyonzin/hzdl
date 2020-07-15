#include "input.h"

void Input(dnn* net, int n, int c, int h, int w) {
    layer* l = (layer*) malloc(sizeof(layer));
    l->dnn = net;
    l->n = n;
    l->c = c;
    l->h = h;
    l->w = w;
    l->type = layer_type_input;

    l->forward = NULL;
    l->backward = NULL;
    l->update_weight = NULL;
    l->destroy = InputDestroy;

    l->in = NULL;
    l->weight = NULL;
    l->bias = NULL;
    l->delta = NULL;
    l->out = (float*) malloc(n * c * h * w * sizeof(float));

    l->next = NULL;
    l->prev = NULL;
    net->edge = net->next = l;
}

void InputDestroy(layer* l) {
    _safe_free(&l->out);
}

