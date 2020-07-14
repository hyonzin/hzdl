#include "softmax.h"

void Softmax(dnn* net) {
    int i;
    int dim = net->edge->c * net->edge->h * net->edge->w;

    layer* l = (layer*) malloc(sizeof(layer));
    l->dnn = net;
    l->n = net->edge->n;
    l->c = 1;
    l->h = 1;
    l->w = dim;
    l->type = layer_type_softmax;
   
    l->activation = NULL;
    l->forward = SoftmaxForward;
    l->backward = SoftmaxBackward;
    
    l->in = net->edge->out;
    l->weight = NULL;
    l->bias = NULL;
    
    // Malloc for output
    l->out = (float*) malloc((l->n * dim) * sizeof(float));

    l->next = NULL;
    l->prev = net->edge;
    net->edge->next = l;
    net->edge = l;
}

void SoftmaxForward(layer* p) {
    int n, i;
    int dim = p->c * p->h * p->w;
    
    for (n=0; n<p->n; ++n) {
        float *in, *out;
        float sum = 0;

        in = &p->in[n * dim];
        out = &p->out[n * dim];

        // Sum of exponential
        for (i=0; i<dim; ++i) {
            sum += exp(in[i]);
        }
        
        // Calculate ratio
        for (i=0; i<dim; ++i) {
            out[i] = exp(in[i]) / sum;
        }
    }
}

void SoftmaxBackward(layer* p) {
    int n, dim, d;
    float sum;
    float *in, *out;

    dim = p->c * p->h * p->w;

    for (n=0; n<p->n; ++n) {
        in = &p->out[n*dim];
        out = &p->in[n*dim];
        
        sum = 0;
        for (d=0; d<dim; ++d) {
            sum += in[d];
        }
        
        for (d=0; d<dim; ++d) {
            out[d] = (in[d] * (sum - in[d])) / (sum * sum);
        }
    }
}

