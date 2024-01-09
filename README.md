# IMAX_GNN
## 概要
IMAXでGNNを実装するプロジェクト。

## コンパイル方法
GCN
```shell
make imax_gcn
```
SpMMテスト
```shell
make test_sparse
```
MMテスト
```shell
make test_dense
```
EMAX7用はターゲットに.emax7や.emax7+dmaを追加すればコンパイルできる。EMAX6も同様。
ただし、EMAX6はオプションとして`EMAX_VER=6`を追加する必要がある。

なお、他アーキテクチャの場合、
- CPU(OpenMP): `CPUONLY=1`
- GPU(CUDA): `CUDA=1`
と設定する。

## IMAX形式への変換
そのままでは行列が大きすぎるため、疎行列も密行列も前処理として分割を行っている。
変換関数内で実装されているため、`utils.h`内の関数を用いて変換することができる。

### `imax_sparse_allocation(IMAXSparseMatrix*)`
メモリ割当を行う。IMAX形式に変換されたものを移し替えているだけなので、予め下記の関数を用いて変換を行う必要がある。

### `convert_imax_sparse_matrix(IMAXSparseMatrix*,SparseMatrix*)`
SparseMatrixをIMAXSparseMatrixに変換する関数。
ホストにメモリ割り当てを行い、IMAXSparseMatrixのデータを格納する。

### `imax_dense_allocation(IMAXDenseMatrix*)`
メモリ割当を行う。

### `convert_imax_dense_matrix(IMAXDenseMatrix*,DenseMatrix*)`
DenseMatrixをIMAXDenseMatrixに変換する関数。
移し替えのみなので、先にIMAXにメモリを割当ておいてから呼び出すこと。

### `convert_dense_matrix(IMAXDenseMatrix*,DenseMatrix*)`
IMAXDenseMatrixをDenseMatrixに変換する関数。
移し替えのみなので、先にIMAXにメモリを割当ておいてから呼び出すこと。

### `imax_sparse_format_init(IMAXSparseMatrix*,int,int,int,int)`
IMAXSparseMatrixを初期化する関数。後述の関数では初期化できないので、この関数を用いて初期化する必要がある。

### `imax_matrix_init_spmm(IMAXDenseMatrix*,IMAXSparseMatrix*,IMAXDenseMatrix*)`
SpMM演算を行うIMAXDenseMatrixを初期化する関数。
メモリ割り当ては行われないので、あらかじめメモリを割り当てておく必要がある。

### `imax_matrix_init_mm(IMAXDenseMatrix*,IMAXDenseMatrix*,IMAXDenseMatrix*)`
MM演算を行うIMAXDenseMatrixを初期化する関数。
メモリ割り当ては行われないので、あらかじめメモリを割り当てておく必要がある。

## カーネル
基本46個または20個の疎行列の列データを持ってきて実行する形になっている。
IMAX的なSIMD実装になっているため多少特殊な行列の変換を前処理として行わなければならない。
IMAX2においてのGCN向けの要素数が多いときに最適化されたカーネルになっている。

### `spmm(IMAXDenseMatrix*,IMAXSparseMatrix*,IMAXDenseMatrix*)`
SpMMを行うカーネル。

### `mm(IMAXDenseMatrix*,IMAXDenseMatrix*,IMAXDenseMatrix*)`
MMを行うカーネル。

## 本プログラムの作成意義
- 巨大行列のSpMMをIMAX2で効率的に動かすことが可能かの検証
- 分割実行時のボトルネック調査
- さらに必要なハードウェア的改造事項の洗い出し