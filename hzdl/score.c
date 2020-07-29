#include "hzdl/score.h"


float Accuracy(dnn* net, float* labels) {
    int out_dim = _get_num_element(net->edge);
    float score = 0;

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
            score+=1;
        }
    }
    return score;
}

float Loss(dnn* net, float* labels) {
    int out_dim = _get_num_element(net->edge);
    int batch_size = net->next->n;
    float loss = 0;

    for (int i=0; i < batch_size; ++i) {
        for (int j=0; j < out_dim; ++j) {
            float val = net->edge->out[i*out_dim + j] - labels[i*out_dim + j];
            loss += (val * val);
        }
    }
    return loss;
}

