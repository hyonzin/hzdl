#include <stdio.h>
#include "hzdl/dnn.h"
#include "hzdl/read_mnist.h"


int example_mnist(int argc, char* argv[]) {
    float* train_images = read_mnist_train_images();
    float* train_labels = read_mnist_train_labels();
    float* test_images = read_mnist_test_images();
    float* test_labels = read_mnist_test_labels();
    int train_size = 60000, test_size = 10000;
    int batch_size = 128, epochs = 100;

    show_mnist(train_labels, train_images, 59999);

    dnn* net;
    CreateDNN(&net);

    Input(net, batch_size, 1, 28, 28);
    Dense(net, 256, Sigmoid);
    Dense(net, 10, Sigmoid);
    Softmax(net);
    
    Train(net, train_images, train_labels, train_size, batch_size, epochs);

    DestroyDNN(&net);

    free(train_images);
    free(train_labels);
    free(test_images);
    free(test_labels);

    return 0;
}

int main(int argc, char* argv[]) {
    return example_mnist(argc, argv);
}
