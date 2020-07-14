#include "read_mnist.h"


float* read_mnist_train_images() {
    int i;
    float* buf = (float*) malloc(60000 * 28 * 28 * sizeof(float));
    unsigned char* cbuf = (unsigned char*) malloc(60000 * 28 * 28);
    FILE* fp = fopen("dataset/mnist/train-images-idx3-ubyte", "r");
    fseek(fp, 16, SEEK_SET);
    fread(cbuf, 1, 60000 * 28 * 28, fp);
    fclose(fp);

    for (i=0; i<60000 * 28 * 28; ++i) {
        buf[i] = (float)cbuf[i] / 255.0;
    }

    return buf;
}

float* read_mnist_train_labels() {
    int i;
    float* buf = (float*) malloc(60000 * sizeof(float));
    unsigned char* cbuf = (unsigned char*) malloc(60000);
    FILE* fp = fopen("dataset/mnist/train-labels-idx1-ubyte", "r");
    fseek(fp, 8, SEEK_SET);
    fread(cbuf, 1, 60000, fp);
    fclose(fp);

    for (i=0; i<60000; ++i) {
        buf[i] = (float)cbuf[i];
    }

    return buf;
}

float* read_mnist_test_images() {
    int i;
    float* buf = (float*) malloc(10000 * 28 * 28 * sizeof(float));
    unsigned char* cbuf = (unsigned char*) malloc(10000 * 28 * 28);
    FILE* fp = fopen("dataset/mnist/t10k-images-idx3-ubyte", "r");
    fseek(fp, 16, SEEK_SET);
    fread(cbuf, 1, 10000 * 28 * 28, fp);
    fclose(fp);

    for (i=0; i<10000*28*28; ++i) {
        buf[i] = (float)cbuf[i] / 255.0;
    }

    return buf;
}

float* read_mnist_test_labels() {
    int i;
    float* buf = (float*) malloc(10000 * sizeof(float));
    unsigned char* cbuf = (unsigned char*) malloc(10000);
    FILE* fp = fopen("dataset/mnist/t10k-labels-idx1-ubyte", "r");
    fseek(fp, 8, SEEK_SET);
    fread(cbuf, 1, 10000, fp);
    fclose(fp);

    for (i=0; i<10000; ++i) {
        buf[i] = (float)cbuf[i];
    }

    return buf;
}

void test_mnist(float* label, float* image, int idx) {
    int i, j;
    printf("label: %.0f\n", label[idx]);

    for (i=0; i<28; ++i) {
        for (j=0; j<28; ++j) {
            if (image[(28*28*idx) + i*28+j] > 255/2) printf("#");
            else printf(" ");
        }
        printf("\n");
    }
    printf("\n");
}

void feed_mnist_images(float* dst, unsigned char* src, int offset, int num) {
    int i, j, k;
    int m = 0;

    src = &src[(offset * 28 * 28)];
    for (i=0; i<num; ++i) {
        for (j=0; j<28; ++j) {
            for (k=0; k<28; ++k) {
                dst[m] = (float)src[m];
                ++m;
            }
        }
    }
}

void feed_mnist_labels(float* dst, unsigned char* src, int offset, int num) {
    int i, j;
    int m = 0, n = 0;

    src = &src[offset];
    for (i=0; i<num; ++i) {
        for (j=0; j<10; ++j) {
            if (j == src[i])
                dst[i*10 + j] = 1;
            else
                dst[i*10 + j] = 0;
        }
    }
}
