// EMAX6/7 GCN Test Program            //
// gcn.h                               //
//         Copyright (C) 2023 by NAIST //
//          Primary writer: Dohyun Kim //
//          kim.dohyun.kg7@is.naist.jp //
#ifndef __GCN_H__
#define __GCN_H__
#if defined(EMAX7)
#include "../conv-c2d/emax7.h"
#elif defined(EMAX6)
#include "../conv-c2c/emax6.h"
#endif
#include "sparse.h"
#include "layer.h"

typedef struct gcn_layer {
    HiddenLayer    hidden_layer; 
    HiddenLayer  latent_vectors;
    HiddenLayer    result_layer;
    struct gcn_layer      *prev;
    struct gcn_layer      *next;
} GCNLayer;

typedef struct gcn_network {
    int     num_layers;
    SparseGraph *graph;
    GCNLayer   *layers;
} GCNNetwork;

void print_gcn_layers(GCNNetwork *network);
void add_gcn_layer(GCNNetwork *network, DenseMatrix weight, DenseMatrix vectors);
void gcn_propagation(GCNNetwork *network);

#endif