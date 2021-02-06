#include "hzdl/metric.h"

void GetMetricName(char* name,
        float (*metric_function)(struct _dnn*, float*)) {
    if (metric_function == Accuracy) {
        sprintf(name, "Accuracy");
    } else if (metric_function == Loss) {
        sprintf(name, "Loss");
    } else {
        sprintf(name, "Unknown");
    }
}

float Accuracy(dnn* net, float* labels) {
    int out_dim = _get_num_element(net->edge);
    float metric = 0;

    for (int i=0; i < net->next->n; ++i) {
        int label = -1;
        int max_idx = -1;
        float max_val = -1;
        for (int j=0; j < out_dim; ++j) {
            float val = net->edge->out[i*out_dim + j];
            if (val > max_val) {
                max_idx = j;
                max_val = val;
            }
            if (labels[i*out_dim + j] == 1) {
                label = j;
            }
        }

        if (max_idx == label) {
            metric+=1;
        }
    }
    return metric;
}

float Loss(dnn* net, float* labels) {
    int out_dim = _get_num_element(net->edge);
    int batch_size = net->next->n;
    float loss = 0;

    for (int i=0; i < batch_size; ++i) {
        for (int j=0; j < out_dim; ++j) {
            float val = net->edge->out[i*out_dim + j]
                - labels[i*out_dim + j];
            loss += (val * val);
        }
    }
    return loss;
}

