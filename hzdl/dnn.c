#include "hzdl/dnn.h"

void CreateDNN(dnn** net) {
    if (net == NULL) return;

    dnn* new_net = (dnn*) malloc(sizeof(dnn));
    new_net->next = NULL;
    new_net->edge = NULL;
    *net = new_net;
}

void DestroyDNN(dnn** net) {
    layer *l, *prev;

    if (net != NULL && *net != NULL && (*net)->next != NULL) {
        l = (*net)->next;
        while (l) {
            prev = l;
            l = l->next;
            prev->destroy(prev);
            free(prev);
        }
    }

    if (net != NULL && *net != NULL) {
        free(*net);
        *net = NULL;
    }
}

void Forward(dnn* net) {
    layer* l = net->next;
    if (l == NULL) return;
    l = l->next;

    while (l) {
        if (l->forward != NULL)
            l->forward(l);
        l = l->next;
    }
}

void Backward(dnn* net, float* labels) {
    layer* l = net->edge;
    if (l == NULL) return;

    while (l && l->type != layer_type_input) {
        if (l->backward != NULL)
            l->backward(l, labels);
        l = l->prev;
    }
}

void UpdateWeight(dnn* net, float learning_rate) {
    layer* l = net->edge;
    if (l == NULL) return;

    while (l && l->type != layer_type_input) {
        if (l->update_weight != NULL)
            l->update_weight(l, learning_rate);
        l = l->prev;
    }
}

void Train(dnn* net,
        float* train_images, float* train_labels, int train_size,
        float* test_images, float* test_labels, int test_size,
        int epochs, float learning_rate) {
    float* labels;
    int offset;
    int epoch_cnt;
    int batch_size;
    int in_dim, out_dim;
    
    assert(net != NULL && net->next != NULL);

    batch_size = net->next->n;
    in_dim = _get_num_element(net->next);
    out_dim = _get_num_element(net->edge);

    labels = malloc(batch_size * out_dim * sizeof(float));
    epoch_cnt = 0;
    while (epoch_cnt++ < epochs) {
        _time_start();

        offset = 0;
        while (offset + batch_size <= train_size) {
            layer* l = net->next;
            if (l == NULL) break;

            // Feed input data
            memcpy(l->out, train_images + offset * in_dim, batch_size * in_dim * sizeof(float));

            // Set label
            memcpy(labels, train_labels + offset * out_dim, batch_size * out_dim * sizeof(float));

            // Forward
            Forward(net);

            // Backward and update weight
            Backward(net, labels);
            UpdateWeight(net, learning_rate);

            // Add offset for the next batch
            offset += batch_size;
        }
        _time_end();

        printf("epoch %d: %.0f ms (%.0f img/sec)\n",
                epoch_cnt, _get_time(),
                (float)offset / _get_time() * 1000);

        // Test if test data is given
        if (test_images != NULL && test_labels != NULL && test_size > 0) {
            Test(net, test_images, test_labels, test_size);
        }
    }

    free(labels);
}

void Test(dnn* net, float* test_images, float* test_labels, int test_size) {
    int correct = 0;
    int offset = 0;
    int batch_size;
    int in_dim, out_dim;
    
    assert(net != NULL && net->next != NULL);

    batch_size = net->next->n;
    in_dim = _get_num_element(net->next);
    out_dim = _get_num_element(net->edge);

    while (offset + batch_size <= test_size) {
        layer* l = net->next;

        // Feed input data
        memcpy(l->out, test_images + offset * in_dim, batch_size * in_dim * sizeof(float));
    
        // Forward
        Forward(net);

        // Calculate accuracy
        for (int i=0; i < batch_size; ++i) {
            int label = -1;
            int max_idx = -1;
            float max_val = -1;
            for (int j=0; j < out_dim; ++j) {
                float val = net->edge->out[i*out_dim + j];
                if (val > max_val) {
                    max_idx = j;
                    max_val = val;
                }
                if (test_labels[(offset+i)*out_dim + j] == 1) {
                    label = j;
                }
            }

            if (max_idx == label) {
                correct++;
            }
        }

        // Add offset for the next batch
        offset += batch_size;
    }
    printf(" ====> Acc.: %.2f\n", ((float)correct/offset) * 100);
}

