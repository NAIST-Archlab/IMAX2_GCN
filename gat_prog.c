#include "include/layer.h"
#include "include/utils.h"
#include "include/sparse.h"
#include "include/gat.h"
#include "include/imax.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


int main(int argc, char **argv) {
    // read adj, features and weights
    GATGraph *g = malloc(sizeof(GATGraph));

    read_graph(g, argv[1]);

    GATLayer *new_layer = malloc(sizeof(GATLayer));
    GATLayer *new_layer_s = malloc(sizeof(GATLayer));

    read_layer(new_layer, g->neighbor->row_size, argv[2]);
    int iter;
    if (argc > 4) {
        read_layer(new_layer_s, g->neighbor->row_size, argv[3]);
        iter = atoi(argv[4]);
    }
    else {
        iter = atoi(argv[3]);
    }

    for (int i = 0 ; i < iter ; i++) {
        GATLayer_forward(new_layer, g, 1, 1);   
        if (argc > 4) {
            GATLayer_forward(new_layer_s, g, 0, 0);
            matrix_elu(g->newfeatures[0]);
            logsoftmax(g->newfeatures[0]);
        }

    }
    printDMatrix(g->newfeatures[0], "cora_output");
    #if defined(EMAX6) || defined(EMAX7)
        printf("SpMM usec: ARM:%d DRAIN:%d CONF:%d REGV:%d RANGE:%d LOAD:%d EXEC:%d total:%d\n",
            (Uint)(all_nanosec[SPMM][0]/1000/iter),
            (Uint)(all_nanosec[SPMM][1]/1000/iter),
            (Uint)(all_nanosec[SPMM][2]/1000/iter),
            (Uint)(all_nanosec[SPMM][3]/1000/iter),
            (Uint)(all_nanosec[SPMM][4]/1000/iter),
            (Uint)(all_nanosec[SPMM][5]/1000/iter),
            (Uint)(all_nanosec[SPMM][6]/1000/iter),
            (Uint)(all_nanosec[SPMM][7]/1000/iter));

        printf("MM usec: ARM:%d DRAIN:%d CONF:%d REGV:%d RANGE:%d LOAD:%d EXEC:%d total:%d\n",
            (Uint)(all_nanosec[MM][0]/1000/iter),
            (Uint)(all_nanosec[MM][1]/1000/iter),
            (Uint)(all_nanosec[MM][2]/1000/iter),
            (Uint)(all_nanosec[MM][3]/1000/iter),
            (Uint)(all_nanosec[MM][4]/1000/iter),
            (Uint)(all_nanosec[MM][5]/1000/iter),
            (Uint)(all_nanosec[MM][6]/1000/iter),
            (Uint)(all_nanosec[MM][7]/1000/iter));

        printf("Attention usec: ARM:%d DRAIN:%d CONF:%d REGV:%d RANGE:%d LOAD:%d EXEC:%d total:%d\n",
            (Uint)(all_nanosec[ATTENTION][0]/1000/iter),
            (Uint)(all_nanosec[ATTENTION][1]/1000/iter),
            (Uint)(all_nanosec[ATTENTION][2]/1000/iter),
            (Uint)(all_nanosec[ATTENTION][3]/1000/iter),
            (Uint)(all_nanosec[ATTENTION][4]/1000/iter),
            (Uint)(all_nanosec[ATTENTION][5]/1000/iter),
            (Uint)(all_nanosec[ATTENTION][6]/1000/iter),
            (Uint)(all_nanosec[ATTENTION][7]/1000/iter));
            
    #elif defined(USE_MP)
        printf("SpMM usec: total:%d\n", (Uint)(all_nanosec[SPMM]/1000/iter));
        printf("MM usec: total:%d\n", (Uint)(all_nanosec[MM]/1000/iter));
        printf("Attention usec: total:%d\n", (Uint)(all_nanosec[ATTENTION]/1000/iter));
        printf("ELU usec: total:%d\n", (Uint)(all_nanosec[ELU]/1000/iter));
        printf("LogSoftmax usec: total:%d\n", (Uint)(all_nanosec[LOG_SOFTMAX]/1000/iter));
    #endif

    return 0;
}
