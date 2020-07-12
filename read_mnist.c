#include "read_mnist.h"


unsigned char* read_train_images() {
    unsigned char* buf = (unsigned char*) malloc(60000 * 28 * 28);
    FILE* fp = fopen("dataset/mnist/train-images-idx3-ubyte", "r");
    fseek(fp, 16, SEEK_SET);
    fread(buf, 1, 60000 * 28 * 28, fp);
    fclose(fp);
    return buf;
}

unsigned char* read_train_labels() {
    unsigned char* buf = (unsigned char*) malloc(60000);
    FILE* fp = fopen("dataset/mnist/train-labels-idx1-ubyte", "r");
    fseek(fp, 8, SEEK_SET);
    fread(buf, 1, 60000, fp);
    fclose(fp);
    return buf;
}

unsigned char* read_test_images() {
    unsigned char* buf = (unsigned char*) malloc(10000 * 28 * 28);
    FILE* fp = fopen("dataset/mnist/t10k-images-idx3-ubyte", "r");
    fseek(fp, 16, SEEK_SET);
    fread(buf, 1, 10000 * 28 * 28, fp);
    fclose(fp);
    return buf;
}

unsigned char* read_test_labels() {
    unsigned char* buf = (unsigned char*) malloc(10000);
    FILE* fp = fopen("dataset/mnist/t10k-labels-idx1-ubyte", "r");
    fseek(fp, 8, SEEK_SET);
    fread(buf, 1, 10000, fp);
    fclose(fp);
    return buf;
}

void test_mnist(unsigned char* label, unsigned char* image, int idx) {
    int i, j;
    printf("label: %d\n", label[idx]);

    for (i=0; i<28; ++i) {
        for (j=0; j<28; ++j) {
            if (image[(28*28*idx) + i*28+j] > 255/2) printf("#");
            else printf(" ");
        }
        printf("\n");
    }
    printf("\n");
}
