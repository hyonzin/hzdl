#include "hzdl/layer/dense.h"

void Dense(dnn* net, int dim, activation act) {
    layer* l = (layer*) malloc(sizeof(layer));
    l->dnn = net;
    l->buffer_size = net->edge->n;
    l->n = l->buffer_size;
    l->c = 1;
    l->h = 1;
    l->w = dim;
    l->type = layer_type_dense;
  
    l->forward = DenseForward;
    l->backward = DenseBackward;
    l->update_weight = DenseUpdateWeight;
    l->destroy = DenseDestroy;

    l->act.forward = act.forward;
    l->act.backward = act.backward;
   
    l->in = net->edge->out;
    // Malloc for weight and output
    l->weight = (float*) malloc(
        (l->n * _get_num_element(net->edge) * dim) * sizeof(float));
    l->bias = (float*) malloc((l->n * dim) * sizeof(float));
    l->out = (float*) malloc((l->n * dim) * sizeof(float));
    
    l->delta = (float*) malloc((l->n * dim) * sizeof(float));

    l->is_frozen = 0;

    // Set random values
    {
        int i;
        srand(time(NULL));
        for (i=0; i<l->n * _get_num_element(net->edge) * dim; ++i) {
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

void DenseForward(layer* l) {
    int in_dim = _get_num_element(l->prev);
    int out_dim = _get_num_element(l);

    #pragma omp parallel for
    for (int n=0; n < l->n; ++n) {
        float *in, *out;
        in = &l->in[n * in_dim];
        out = &l->out[n * out_dim];

        for (int o=0; o < out_dim; ++o) {
            float sum = 0;

            // Dot product
            for (int i=0; i < in_dim; ++i) {
                sum += in[i] * l->weight[o * in_dim + i];
            }

            //FIXME bias?
            //sum += l->bias[o];
            out[o] = sum;
        }
    }

    // Activation function
    if (l->act.forward != NULL) {
        #pragma omp parallel for
        for (int n=0; n < l->n; ++n) {
            l->delta[n] = -1;
            float *out = &l->out[n * out_dim];
            for (int o=0; o < out_dim; ++o) {
                out[o] = l->act.forward(l, n, out[o]);
            }
        }
    }
}

void DenseBackward(layer* l, float* labels) {
    int is_last_dense_layer = 0;
    int dim;

    if (l->is_frozen) return;

    if (l->next == NULL) {
        is_last_dense_layer = 1;
    }

    dim = _get_num_element(l);
    
    #pragma omp parallel for
    for (int n=0; n < l->n; ++n) {
        if (l->act.backward) {
            for (int d=0; d < dim; ++d) {
                l->delta[n*dim + d] =
                        l->act.backward(l, n, l->out[n*dim + d]);
            }
        } else {
            for (int d=0; d < dim; ++d) {
                l->delta[n*dim + d] = 1;
            }
        }
    }
    
    if (is_last_dense_layer) {
        #pragma omp parallel for
        for (int n=0; n < l->n; ++n) {
            for (int d=0; d < dim; ++d) {
                l->delta[n*dim + d] *=
                    (l->out[n*dim + d] - labels[n*dim + d]);
            }
        }
    } else {
        int next_dim = _get_num_element(l->next);

        #pragma omp parallel for
        for (int n=0; n < l->n; ++n) {
            for (int d=0; d < dim; ++d) {
                int k;
                float sum = 0;

                for (k=0; k < next_dim; ++k) {
                    sum += l->next->delta[n*next_dim + k]
                        * l->next->weight[k*dim + d];
                }
                l->delta[n*dim + d] *= sum;
            }
        }
    }
}

void DenseUpdateWeight(layer* l, float eta) {
    layer* p = l->prev;

    int dim = _get_num_element(l);
    int prev_dim = _get_num_element(p);

    #pragma omp parallel for
    for (int j=0; j <dim; ++j) {
        for (int i=0; i < prev_dim; ++i) {
            float dw = 0;
            for (int n=0; n < l->n; ++n) {
                dw += p->out[n*prev_dim + i] * l->delta[n*dim + j];
            }
            dw = -eta * dw;

            l->weight[j * prev_dim + i] += dw;
        }
    }
}

void DenseDestroy(layer* l) {
    _safe_free(&l->weight);
    _safe_free(&l->bias);
    _safe_free(&l->out);
    _safe_free(&l->delta);
    if (l!=NULL) free(l);
}

