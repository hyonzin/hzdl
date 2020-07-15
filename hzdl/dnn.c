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
            p = p->next;
            prev->destroy(prev);
            free(prev);
        }
    }

    if (net != NULL && *net != NULL) {
        free(*net);
        *net = NULL;
    }
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

void Backward(dnn* net, int batch_size, float learning_rate, float* labels) {
    layer* p = net->edge;
    if (p == NULL) return;

    while (p && p->type != layer_type_input) {
        if (p->backward != NULL)
            p->backward(p);
        p = p->prev;
    }
}

void Train(dnn* net, float* train_images, float* train_labels,
        int train_size, int batch_size, int epochs, float learning_rate) {
    int offset;
    float* labels;
    int epoch_cnt = 0;
    
    assert(net->edge->n >= batch_size);
    labels = malloc(batch_size * net->edge->c
            * net->edge->h * net->edge->w * sizeof(float));

    while (epoch_cnt++ < epochs) {
        offset = 0;
        while (offset + batch_size <= train_size) {
            layer* p = net->next;
            if (p == NULL) break;

            // Feed input data
            memcpy(p->out, train_images + offset * _get_num_element(p),
                    batch_size * _get_num_element(p));

            // Set label
            memcpy(labels, train_labels + offset,
                    batch_size);

            // Forward up to the last layer
            Forward(net, batch_size);

            // Backward
            Backward(net, batch_size, learning_rate, labels);

            // Add offset for the next batch
            offset += batch_size;

            /////(Debug) just 1 iteration
            {
                int i;
                printf("epoch %d\t", epoch_cnt);
                for (i = 0; i < 10; ++i) {
                    printf("%.4f ", net->edge->out[i]);
                }
                printf("\n");
            }
            free(labels);
            return;
            /////(Debug)
        }
    }

    free(labels);
}

