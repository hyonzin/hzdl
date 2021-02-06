#include "hzdl/file.h"

/*
void SaveStructure(dnn* net, char* filename) {
    FILE* fp;
    layer* l;
    char structure_filename[256];

    if (net == NULL || net->next == NULL || filename == NULL)
        return;

    sprintf(structure_filename, "%s.structure.bin", filename);

    fp = fopen(structure_filename, "w");
    if (!fp) {
        err(-1, "Can't write %s", structure_filename);
    }

    l = net->next;
    while (l) {
        // save type
        l = l->next;
    }

    fclose(fp);
}
*/

void SaveWeight(dnn* net, char* filename) {
    FILE* fp;
    layer* l;
    char weight_filename[256];

    if (net == NULL || net->next == NULL || filename == NULL)
        return;

    sprintf(weight_filename, "%s.weight.bin", filename);

    fp = fopen(weight_filename, "w");
    if (!fp) {
        err(-1, "Can't write %s", weight_filename);
    }

    l = net->next;
    while (l) {
        // save weight
        if (l->weight) {
            fwrite(l->weight, sizeof(float),
                    l->weight_size, fp);
        }
        
        // save bias
        if (l->bias) {
            fwrite(l->bias, sizeof(float),
                    l->bias_size, fp);
        }

        l = l->next;
    }

    fclose(fp);
}

void SaveDNN(dnn* net, char* filename) {
    // SaveStructure(net, filename);
    SaveWeight(net, filename);
}

/*
void LoadStructure(dnn** net, char* filename) {
    FILE* fp;
    layer* l;
    char structure_filename[256];

    if (net == NULL || *net == NULL ||  (*net)->next == NULL || filename == NULL)
        return;

    sprintf(structure_filename, "%s.structure.bin", filename);

    fp = fopen(structure_filename, "r");
    if (!fp) {
        err(-1, "Can't read %s", structure_filename);
    }

    l = (*net)->next;
    while (l) {
        l = l->next;
    }

    fclose(fp);
}
*/

void LoadWeight(dnn** net, char* filename) {
    FILE* fp;
    layer* l;
    char weight_filename[256];

    if (net == NULL || *net== NULL || (*net)->next == NULL || filename == NULL)
        return;

    sprintf(weight_filename, "%s.weight.bin", filename);

    fp = fopen(weight_filename, "r");
    if (!fp) {
        err(-1, "Can't read %s", weight_filename);
    }

    l = (*net)->next;
    while (l) {
        int sz;
        // load weight
        if (l->weight) {
            sz = fread(l->weight, sizeof(float),
                    l->weight_size, fp);
        }
        
        // load bias
        if (l->bias) {
            sz = fread(l->bias, sizeof(float),
                    l->bias_size, fp);
        }

        l = l->next;
    }

    fclose(fp);
}

void LoadDNN(dnn** net, char* filename) {
    // LoadStructure(net, filename);
    LoadWeight(net, filename);
}

