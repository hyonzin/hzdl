#include <stdio.h>
#include "hzdl/dnn.h"
#include "hzdl/dataset/mnist.h"

#define MNIST_DIR "data/mnist"

#define ActFunc None

float LossAndShow(dnn* net, float* labels) {
    if (!net->is_training) {
        float tr[10] = {0, 1,};
        float te[10] = {1, 0,};
        show_mnist(tr, net->next->out, 0);
        show_mnist(te, net->edge->out, 0);
    }
    return Loss(net, labels);
}


int test_mnist(int argc, char* argv[]) {
    float* train_images = read_mnist_train_images(MNIST_DIR);
    float* test_images = read_mnist_test_images(MNIST_DIR);

    int train_size = 60000, test_size = 10000;
    int batch_size = 64, epochs = 30;
    float learning_rate = 0.0001;

    dnn* net;
    CreateDNN(&net);

    Input(net, batch_size, 1, 28, 28);
    Dense(net, 128, ActFunc);
    Dense(net, 64, ActFunc);
    Dense(net, 128, ActFunc);
    Dense(net, 28*28, ActFunc);
   
    Train(net, train_images, train_images, train_size,
            test_images, test_images, test_size,
            epochs, batch_size, learning_rate,
            // LossAndShow);
            Loss);

    DestroyDNN(&net);

    free(train_images);
    free(test_images);

    return 0;
}

int main(int argc, char* argv[]) {
    return test_mnist(argc, argv);
}

