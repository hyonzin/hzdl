#include <stdio.h>
#include "hzdl/dnn.h"
#include "hzdl/example/mnist.h"

#define MNIST_DIR "dataset/mnist"

int example_mnist(int argc, char* argv[]) {
    float* train_images = read_mnist_train_images(MNIST_DIR);
    float* train_labels = read_mnist_train_labels(MNIST_DIR);
    float* test_images = read_mnist_test_images(MNIST_DIR);
    float* test_labels = read_mnist_test_labels(MNIST_DIR);

    int train_size = 60000, test_size = 10000;
    int batch_size = 256, epochs = 100;
    float learning_rate = 0.1;

    //show_mnist(train_labels, train_images, 59999);

    dnn* net;
    CreateDNN(&net);

    Input(net, batch_size, 1, 28, 28);
    Dense(net, 256, Sigmoid);
    Dense(net, 10, Softmax);
    
    Train(net, train_images, train_labels, train_size,
            test_images, test_labels, test_size,
            epochs, learning_rate);

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
