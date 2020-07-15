#include "dense.h"

void Dense(dnn* net, int dim, activation act) {
    layer* l = (layer*) malloc(sizeof(layer));
    l->dnn = net;
    l->n = net->edge->n;
    l->c = 1;
    l->h = 1;
    l->w = dim;
    l->type = layer_type_dense;
  
    l->forward = DenseForward;
    l->backward = DenseBackward;
    l->destroy = DenseDestroy;

    l->act.forward = act.forward;
    l->act.backward = act.backward;
   
    l->in = net->edge->out;
    // Malloc for weight and output
    l->weight = (float*) malloc(
            (l->n * (net->edge->c * net->edge->h * net->edge->w) * dim)
            * sizeof(float));
    l->bias = (float*) malloc((l->n * dim) * sizeof(float));
    l->out = (float*) malloc((l->n * dim) * sizeof(float));
    
    l->delta = (float*) malloc((l->n * dim) * sizeof(float)); //FIXME no need if it's not training

    // Set random values
    {
        int i;
        srand(time(NULL));
        for (i=0; i<l->n * (net->edge->c * net->edge->h * net->edge->w) * dim; ++i) {
            l->weight[i] = _random_float();
        }
        for (i=0; i<l->n * dim; ++i) {
            l->bias[i] = _random_float();
        }
    }

    l->next = NULL;
    l->prev = net->edge;
    net->edge->next = l;
    net->edge = l;
}

void DenseForward(layer* p) {
    int n, i, o;
    int in_dim = p->prev->c * p->prev->h * p->prev->w;
    int out_dim = p->c * p->h * p->w;

    for (n=0; n<p->n; ++n) {
        float *in, *out;
        in = &p->in[n * in_dim];
        out = &p->out[n * out_dim];

        for (o=0; o<p->c * p->h * p->w; ++o) {
            float sum = 0;

            // Dot product
            for (i=0; i<in_dim; ++i) {
                sum += in[i] * p->weight[o * in_dim + i];
            }
            sum += p->bias[o];

            // Activation
            if (p->act.forward != NULL) {
                sum = p->act.forward(sum);
            }

            out[o] = sum;
        }
    }
}

void DenseBackward(layer* p) {
    ;
}

void DenseDestroy(layer* p) {
    _safe_free(&p->weight);
    _safe_free(&p->bias);
    _safe_free(&p->out);
    _safe_free(&p->delta); //FIXME no need if it's not training
}

