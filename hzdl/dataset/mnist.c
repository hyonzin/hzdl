#include "hzdl/dataset/mnist.h"


float* read_mnist_train_images(char* dir) {
    int i, res;
    float* buf = (float*) malloc(60000 * 28 * 28 * sizeof(float));
    unsigned char* cbuf = (unsigned char*) malloc(60000 * 28 * 28);
    char path[256];
    FILE* fp;
    
    sprintf(path, "%s/train-images-idx3-ubyte", dir);
    fp = fopen(path, "r");
    fseek(fp, 16, SEEK_SET);
    res = fread(cbuf, 1, 60000 * 28 * 28, fp);
    if (res <= 0) {
        printf("read failed(%s)\n", path);
        return NULL;
    }
    fclose(fp);

    for (i=0; i<60000 * 28 * 28; ++i) {
        buf[i] = (float)cbuf[i] / 255.0;
    }

    return buf;
}

float* read_mnist_train_labels(char* dir) {
    int i, j, res;
    float* buf = (float*) malloc(60000 * 10 * sizeof(float));
    unsigned char* cbuf = (unsigned char*) malloc(60000);
    char path[256];
    FILE* fp;
    
    sprintf(path, "%s/train-labels-idx1-ubyte", dir);
    fp = fopen(path, "r");
    fseek(fp, 8, SEEK_SET);
    res = fread(cbuf, 1, 60000, fp);
    if (res <= 0) {
        printf("read failed(%s)\n", path);
        return NULL;
    }
    fclose(fp);

    for (i=0; i<60000; ++i) {
        for (j=0; j<10; ++j) {
            if (j == cbuf[i])
                buf[i*10 + j] = 1;
            else
                buf[i*10 + j] = 0;
        }
    }

    return buf;
}

float* read_mnist_test_images(char* dir) {
    int i, res;
    float* buf = (float*) malloc(10000 * 28 * 28 * sizeof(float));
    unsigned char* cbuf = (unsigned char*) malloc(10000 * 28 * 28);
    char path[256];
    FILE* fp;
    
    sprintf(path, "%s/t10k-images-idx3-ubyte", dir);
    fp = fopen(path, "r");
    fseek(fp, 16, SEEK_SET);
    res = fread(cbuf, 1, 10000 * 28 * 28, fp);
    if (res <= 0) {
        printf("read failed(%s)\n", path);
        return NULL;
    }
    fclose(fp);

    for (i=0; i<10000*28*28; ++i) {
        buf[i] = (float)cbuf[i] / 255.0;
    }

    return buf;
}

float* read_mnist_test_labels(char* dir) {
    int i, j, res;
    float* buf = (float*) malloc(10000 * 10 * sizeof(float));
    unsigned char* cbuf = (unsigned char*) malloc(10000);
    char path[256];
    FILE* fp;
    
    sprintf(path, "%s/t10k-labels-idx1-ubyte", dir);
    fp = fopen(path, "r");
    fseek(fp, 8, SEEK_SET);
    res = fread(cbuf, 1, 10000, fp);
    if (res <= 0) {
        printf("read failed(%s)\n", path);
        return NULL;
    }
    fclose(fp);

    for (i=0; i<10000; ++i) {
        for (j=0; j<10; ++j) {
            if (j == cbuf[i])
                buf[i*10 + j] = 1;
            else
                buf[i*10 + j] = 0;
        }
    }

    return buf;
}

void show_mnist(float* label, float* image, int idx) {
    int i, j;
    char* pen = " .:-=+*#%@";

    printf("label: ");
    for (i=0; i<10; ++i) {
        if (label[idx*10 + i] > 0) {
            printf("%d\n", i);
            break;
        }
    }

    for (i=0; i<28; ++i) {
        for (j=0; j<28; ++j) {
            int val = (image[(28*28*idx) + i*28+j] * 10);
            val = fmax(fmin(val, 9), 0);
            printf("%c", pen[val]);
        }
        printf("\n");
    }
    printf("\n");
}

//FIXME
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

//FIXME
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
