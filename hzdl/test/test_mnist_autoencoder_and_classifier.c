#include <stdio.h>
#include "hzdl/dnn.h"
#include "hzdl/dataset/mnist.h"

#define MNIST_DIR "data/mnist"

/* MNIST Transfer Learning
 * (AutoEncoder & Classifier) */

int test_mnist(int argc, char* argv[]) {
    float* train_images = read_mnist_train_images(MNIST_DIR);
    float* train_labels = read_mnist_train_labels(MNIST_DIR);
    float* test_images = read_mnist_test_images(MNIST_DIR);
    float* test_labels = read_mnist_test_labels(MNIST_DIR);

    int train_size = 60000, test_size = 10000;
    int batch_size = 64;
    float epochs, lr;

    dnn* net;
    CreateDNN(&net);

    Input(net, batch_size, 1, 28, 28);
    Dense(net, 128, None);
    Dense(net, 64, None);
    Dense(net, 128, None);
    Dense(net, 28*28, None);

    // AutoEncoder unsupervised learning
    epochs = 30;
    lr = 0.0001;
    Train(net, train_images, train_images, train_size,
            test_images, test_images, test_size,
            epochs, batch_size, lr,
            Loss);

    DeleteLastLayer(net);
    DeleteLastLayer(net);

    Freeze(net);
    Dense(net, 10, Softmax);

    // Classifier supervised fine-tuning
    epochs = 15;
    lr = 0.01;
    Train(net, train_images, train_labels, train_size,
            test_images, test_labels, test_size,
            epochs, batch_size, lr,
            Accuracy);

    DestroyDNN(&net);

    free(train_images);
    free(train_labels);
    free(test_images);
    free(test_labels);

    return 0;
}

int main(int argc, char* argv[]) {
    return test_mnist(argc, argv);
}

