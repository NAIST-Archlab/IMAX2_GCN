// EMAX6/7 GCN Test Program            //
// reader.h                            //
//         Copyright (C) 2023 by NAIST //
//          Primary writer: Dohyun Kim //
//          kim.dohyun.kg7@is.naist.jp //
#ifndef __READER_H__
#define __READER_H__
#include "layer.h"
#include "sparse.h"

void read_graph_bin(SparseGraph *g, char *name, int from, int to);
void read_graph_csgr(SparseGraph *g, char *name, int from, int to);
void read_gcn_weight(GCNNetwork *n, char *path);
void read_gcn_feature_bin(GCNNetwork *n, char *name, int from, int to);
void read_gcn_feature_csgr(GCNNetwork *n, char *name, int from, int to);
#endif