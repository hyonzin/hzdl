#include <stdio.h>
#include "read_mnist.h"


int main(int argc, char* argv[]) {
    unsigned char* train_images = read_train_images();
    unsigned char* train_labels = read_train_labels();
    unsigned char* test_images = read_test_images();
    unsigned char* test_labels = read_test_labels();

    test_mnist(train_labels, train_images, 10000);

    free(train_images);
    free(train_labels);
    free(test_images);
    free(test_labels);

    return 0;
}
