#include "dnn.h"
#include "layer/layers.h"

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
    l->activation = NULL;
    l->forward = NULL;
    l->backward = NULL;
    l->in = NULL;
    l->weight = NULL;
    l->bias = NULL;
    l->out = (float*) malloc(n * c * h * w * sizeof(float));

    l->next = NULL;
    l->prev = NULL;
    net->edge = net->next = l;
}

void Forward(dnn* net, int batch_size) {
    layer* p = net->next;
    if (p == NULL) return;
    p = p->next;

    while (p) {
        if (p->forward != NULL)
            p->forward(p);
        p = p->next;
    }
}

void Backward(dnn* net, int batch_size, float* labels) {
    layer* p = net->edge;
    if (p == NULL) return;

    while (p && p->type != layer_type_input) {
        if (p->backward != NULL)
            p->backward(p);
        p = p->prev;
    }
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

