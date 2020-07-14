#include "dnn.h"

void CreateDNN(dnn** net) {
    if (net == NULL) return;

    dnn* new_net = (dnn*) malloc(sizeof(dnn));
    new_net->next = NULL;
    new_net->edge = NULL;
    *net = new_net;
}

void DestroyDNN(dnn** net) {
    layer *p, *prev;

    if (net != NULL && *net != NULL && (*net)->next != NULL) {
        p = (*net)->next;
        while (p) {
            prev = p;
            if (p->out != NULL)
                free(p->out);
            if (p->weight != NULL)
                free(p->weight);
            if (p->bias != NULL)
                free(p->bias);
            p = p->next;
            free(prev);
        }
    }

    if (net != NULL && *net != NULL) {
        free(*net);
        *net = NULL;
    }
}

void Input(dnn* net, int n, int c, int h, int w) {
    layer* l = (layer*) malloc(sizeof(layer));
    l->dnn = net;
    l->n = n;
    l->c = c;
    l->h = h;
    l->w = w;
    l->type = layer_type_input;
    l->in = NULL;
    l->weight = NULL;
    l->bias = NULL;
    l->out = (float*) malloc(n * c * h * w * sizeof(float));

    l->next = NULL;
    l->prev = NULL;
    net->edge = net->next = l;
}

void Dense(dnn* net, int dim, float (*activation)(float)) {
    layer* l = (layer*) malloc(sizeof(layer));
    l->dnn = net;
    l->n = net->edge->n;
    l->c = 1;
    l->h = 1;
    l->w = dim;
    l->type = layer_type_dense;
    l->in = net->edge->out;
    l->activation = activation;

    // Malloc for weight and output
    l->weight = (float*) malloc(
            (l->n * (net->edge->c * net->edge->h * net->edge->w) * dim)
            * sizeof(float));
    l->bias = (float*) malloc((l->n * dim) * sizeof(float));
    l->out = (float*) malloc((l->n * dim) * sizeof(float));

    // Set random values
    {
        int i;
        srand(time(NULL));
        for (i=0; i<l->n * (net->edge->c * net->edge->h * net->edge->w) * dim; ++i) {
            l->weight[i] = sin((float)rand());
        }
        for (i=0; i<l->n * dim; ++i) {
            l->bias[i] = sin((float)rand());
        }
    }

    l->next = NULL;
    l->prev = net->edge;
    net->edge->next = l;
    net->edge = l;
}

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

void ForwardDense(layer* p) {
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
            if (p->activation != NULL) {
                sum = p->activation(sum);
            }

            out[o] = sum;
        }
    }
}

void ForwardSoftmax(layer* p) {
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

void Forward(dnn* net, int batch_size) {
    layer* p = net->next;
    if (p == NULL) return;
    p = p->next;

    while (p) {
        switch (p->type) {
        case layer_type_dense:
            ForwardDense(p);
            break;
        case layer_type_softmax:
            ForwardSoftmax(p);
            break;
        }

        p = p->next;
    }
}

void Backward(dnn* net, int batch_size, float* labels) {
    layer* p = net->edge;

}

void Train(dnn* net, float* train_images, float* train_labels,
        int train_size, int batch_size, int epochs) {
    int b_size;
    int rest_train_size = train_size;
    float* labels = malloc(net->edge->n * net->edge->c
            * net->edge->h * net->edge->w * sizeof(float));

    while (rest_train_size > 0) {
        layer* p = net->next;
        if (p == NULL) break;
        p = p->next;

        // Set batch size
        b_size = (batch_size < rest_train_size)? batch_size : rest_train_size;
        rest_train_size -= b_size;

        // Feed input data
        memcpy(p->in, train_images, b_size * 28 * 28);

        // Set label
        memcpy(labels, train_labels, b_size);

        // Forward up to the last layer
        Forward(net, b_size);
        
        // Backward
        Backward(net, b_size, labels);

        // Add offset for the next batch
        train_images = train_images + b_size * 28 * 28;
        train_labels = train_labels + b_size;

        /////(Debug) just 1 iteration
        for (b_size = 0; b_size < 10; ++b_size) {
            printf("%.4f\n", net->edge->out[b_size]);
        }
        printf("\n");
        break;
    }

    free(labels);
}

