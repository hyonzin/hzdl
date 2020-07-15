#include "dnn.h"

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
    
    assert(net != NULL && net->next != NULL);

    batch_size = net->next->n;
    labels = malloc(batch_size * net->edge->c
            * net->edge->h * net->edge->w * sizeof(float));

    epoch_cnt = 0;
    while (epoch_cnt++ < epochs) {
        _time_start();

        offset = 0;
        while (offset + batch_size <= train_size) {
            layer* l = net->next;
            if (l == NULL) break;

            // Feed input data
            memcpy(l->out, train_images + offset * _get_num_element(l),
                    batch_size * _get_num_element(l));

            // Set label
            memcpy(labels, train_labels + offset, batch_size);

            // Forward up to the last layer
            Forward(net);

            // Backward
            Backward(net, labels);
            UpdateWeight(net, learning_rate);

            // Add offset for the next batch
            offset += batch_size;
        }
        _time_end();

        {
            printf("..............%.0f => ", labels[0]);
            for(int i=0; i<10; ++i) {
                printf("%.4f ", net->edge->out[i]);
            }
            printf("\n");
        }

        {
            printf("epoch %d: %.0f ms (%.0f img/sec)\n",
                    epoch_cnt, _get_time(),
                    (float)offset / _get_time() * 1000);

            if (test_images != NULL && test_labels != NULL && test_size > 0) {
                int dim = _get_num_element(net->edge);
                int correct = 0;
                offset = 0;
                while (offset + batch_size <= test_size) {
                    layer* l = net->next;

                    // Feed input data
                    memcpy(l->out, train_images + offset * _get_num_element(l),
                            batch_size * _get_num_element(l));
                
                    // Forward up to the last layer
                    Forward(net);

                    // Calculate accuracy
                    for (int i=0; i < batch_size; ++i) {
                        int max_idx = -1;
                        int max_val = -1;
                        for (int j=0; j < dim; ++j) {
                            float val = net->edge->out[i*dim + j];
//                            printf("%.2f, ", val);
                            if (val > max_val) {
                                max_idx = j;
                                max_val = val;
                            }
                        }
//                        printf("%d vs %f\n", max_idx, train_labels[offset + i]);
                        if (max_idx == train_labels[offset + i]) {
                            correct++;
                        }
                    }
            
                    // Add offset for the next batch
                    offset += batch_size;
                }
                printf("  ====> Acc.: %.2f\n", ((float)correct/offset) * 100);
            }
        }
    }

    free(labels);
}

