#include "hzdl/dnn.h"

void CreateDNN(dnn** net) {
    if (net == NULL) return;

    dnn* new_net = (dnn*) malloc(sizeof(dnn));
    new_net->next = NULL;
    new_net->edge = NULL;
    new_net->is_training = 0;
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
        }
    }

    if (net != NULL && *net != NULL) {
        free(*net);
        *net = NULL;
    }
}

void DeleteLastLayer(dnn* net) {
    assert(net != NULL);

    layer* prev;
    layer* l = net->edge;

    if (l != NULL) {
        prev = l->prev;
        l->destroy(l);

        prev->next = NULL;
        net->edge = prev;
    }
}

void Freeze(dnn* net) {
    assert(net != NULL);

    layer* l = net->next;

    while (l && l->type != layer_type_input) {
        l->is_frozen = 1;
        l = l->next;
    }
}

void Melt(dnn* net) {
    assert(net != NULL);

    layer* l = net->next;

    while (l && l->type != layer_type_input) {
        l->is_frozen = 0;
        l = l->next;
    }
}

void UpdateBatchSize(dnn* net, int batch_size) {
    assert(net != NULL);
   
    layer* l = net->next;
    if (l == NULL) return;

    if (batch_size > l->buffer_size) {
        warn("batch size(%d) must be same or less than buffer size(%d)\n",
                batch_size, l->buffer_size);
        batch_size = l->buffer_size;
    }

    while (l) {
        l->n = batch_size;
        l = l->next;
    }
}

void Forward(dnn* net) {
    assert(net != NULL);

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
    assert(net != NULL);

    layer* l = net->edge;

    while (l && l->type != layer_type_input) {
        if (l->backward != NULL)
            l->backward(l, labels);
        l = l->prev;
    }
}

void UpdateWeight(dnn* net, float learning_rate) {
    assert(net != NULL);

    layer* l = net->edge;

    while (l && l->type != layer_type_input) {
        if (l->update_weight != NULL)
            l->update_weight(l, learning_rate);
        l = l->prev;
    }
}

void Train(dnn* net,
        float* train_images, float* train_labels, int train_size,
        float* test_images, float* test_labels, int test_size,
        int epochs, int batch_size, float learning_rate,
        float (*metric_function)(struct _dnn*, float*)) {
    float* labels;
    int epoch_cnt;
    int in_dim, out_dim;
    
    assert(net != NULL);
    assert(net->next != NULL);

    UpdateBatchSize(net, batch_size);

    in_dim = _get_num_element(net->next);
    out_dim = _get_num_element(net->edge);

    epoch_cnt = 0;
    while (epoch_cnt++ < epochs) {
        float metric = 0;
        int offset = 0;

        net->is_training = 1;
        _time_start();

        while (offset + batch_size <= train_size) {
            layer* l = net->next;
            if (l == NULL) break;

            // Feed input data
            memcpy(l->out, train_images + offset * in_dim,
                    batch_size * in_dim * sizeof(float));

            // Set label
            labels = train_labels + offset * out_dim;

            // Forward
            Forward(net);

            // Backward and update weight
            Backward(net, labels);
            UpdateWeight(net, learning_rate);

            // Add offset for the next batch
            offset += batch_size;

            // Calculate metric
            if (metric_function != NULL) {
                metric += metric_function(net, labels);
            }
        }

        _time_end();
        net->is_training = 0;

        printf("epoch %d: %.0f ms (%.0f sample/sec)\n",
                epoch_cnt, _get_time(),
                (float)offset / _get_time() * 1000);

        if (metric_function != NULL) {
            char metric_name[32];
            GetMetricName(metric_name, metric_function);

            printf(" ====> Train %s: %.2f\n",
                    metric_name, ((float)metric/offset));
        }

        // Test if test data is given
        if (metric_function != NULL && test_images != NULL
                && test_labels != NULL && test_size > 0) {
            Test(net, test_images, test_labels, test_size,
                    batch_size, metric_function);
        }
    }
}

void Test(dnn* net, float* test_images, float* test_labels, int test_size,
        int batch_size, float (*metric_function)(struct _dnn*, float*)) {
    float metric = 0;
    int offset = 0;
    int in_dim, out_dim;
    float* labels;
    char metric_name[32];
    
    assert(net != NULL);
    assert(net->next != NULL);
    assert(metric_function != NULL);

    in_dim = _get_num_element(net->next);
    out_dim = _get_num_element(net->edge);

    while (offset + batch_size <= test_size) {
        layer* l = net->next;

        // Feed input data
        memcpy(l->out, test_images + offset * in_dim,
                batch_size * in_dim * sizeof(float));
        
        // Set label
        labels = test_labels + offset * out_dim;
    
        // Forward
        Forward(net);

        // Calculate metric
        metric += metric_function(net, labels);

        // Add offset for the next batch
        offset += batch_size;
    }

    GetMetricName(metric_name, metric_function);
    printf(" ====> Test %s: %.2f\n", metric_name, ((float)metric/offset));
}

