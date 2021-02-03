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
        int epochs, float learning_rate, float (*score_function)(struct _dnn*, float*)) {
    float* labels;
    int epoch_cnt;
    int batch_size;
    int in_dim, out_dim;
    
    assert(net != NULL);
    assert(net->next != NULL);

    batch_size = net->next->n;
    in_dim = _get_num_element(net->next);
    out_dim = _get_num_element(net->edge);

//    labels = malloc(batch_size * out_dim * sizeof(float));
    epoch_cnt = 0;
    while (epoch_cnt++ < epochs) {

        float score = 0;
        int offset = 0;


        net->is_training = 1;
        _time_start();

        while (offset + batch_size <= train_size) {
            layer* l = net->next;
            if (l == NULL) break;

            // Feed input data
            memcpy(l->out, train_images + offset * in_dim, batch_size * in_dim * sizeof(float));

            // Set label
//            memcpy(labels, train_labels + offset * out_dim, batch_size * out_dim * sizeof(float));
            labels = train_labels + offset * out_dim;

            // Forward
            Forward(net);

            // Backward and update weight
            Backward(net, labels);
            UpdateWeight(net, learning_rate);

            // Add offset for the next batch
            offset += batch_size;

            // Calculate score
            if (score_function != NULL) {
                score += score_function(net, labels);
            }
        }

        _time_end();
        net->is_training = 0;

        printf("epoch %d: %.0f ms (%.0f img/sec)\n",
                epoch_cnt, _get_time(),
                (float)offset / _get_time() * 1000);

        if (score_function != NULL) {
            printf(" ====> Train Acc. or Loss: %.2f\n", (score/offset));
        }

        // Test if test data is given
        if (score_function != NULL && test_images != NULL && test_labels != NULL && test_size > 0) {
            Test(net, test_images, test_labels, test_size, score_function);
        }
    }

//    free(labels);
}

void Test(dnn* net, float* test_images, float* test_labels, int test_size, float (*score_function)(struct _dnn*, float*)) {
    float score = 0;
    int offset = 0;
    int batch_size;
    int in_dim, out_dim;
    float* labels;
    
    assert(net != NULL);
    assert(net->next != NULL);
    assert(score_function != NULL);

    batch_size = net->next->n;
    in_dim = _get_num_element(net->next);
    out_dim = _get_num_element(net->edge);

    while (offset + batch_size <= test_size) {
        layer* l = net->next;

        // Feed input data
        memcpy(l->out, test_images + offset * in_dim, batch_size * in_dim * sizeof(float));
        
        // Set label
        labels = test_labels + offset * out_dim;
    
        // Forward
        Forward(net);

        // Calculate score
        score += score_function(net, labels);

        // Add offset for the next batch
        offset += batch_size;
    }

    printf(" ====> Test Acc. or loss: %.2f\n", ((float)score/offset));
}

