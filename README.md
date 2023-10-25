# IMAX_GCN
## 概要
IMAXでGCNを実装するプロジェクト。

## コンパイル方法
本体
```shell
make imax_gcn
```
SpMMテスト
```shell
make test_sparse
```
EMAX7用はターゲットに.emax7や.emax7+dmaを追加すればコンパイルできる。EMAX6も同様。
ただし、EMAX6はオプションとして`EMAX_VER=6`を追加する必要がある。

## 前処理
そのままでは行列が大きすぎるため、疎行列も密行列も前処理として分割を行っている。
変換関数内で実装されているため、`utils.h`内の関数を用いて変換することができる。
なお、IMAX上で動かすために必要なメモリの割当も、当ファイルの`imax_gcn_allocation()`関数で行える。

~~一定の長さに揃える必要があり、効率的なパディングを行うためにはソートが必要であるため、マージソートを前処理として行い、
行列を分割し、実行するためのメモリアドレスに移し、実行を行っている。~~

現在は、マージソートが必要ない仕様になっている。しかし、行列分割及びパディング処理は行っている。

## カーネル
基本46個の疎行列の列データを持ってきて実行する形になっている。
IMAX的なSIMD実装になっているため多少特殊な行列の変換を前処理として行わなければならない。
IMAX2においてのGCN向けの要素数が多いときに最適化されたカーネルになっている。

## 本プログラムの作成意義
- 巨大行列のSpMMをIMAX2で効率的に動かすことが可能かの検証
- 分割実行時のボトルネック調査
- さらに必要なハードウェア的改造事項の洗い出し

## 今後の実装方針
この状態では200万要素を超えるグラフデータの処理は困難であることが判明したため、
IMAX3向けに動くことを前提とし、前処理及び分割を行う方針を維持する。

SpMMの1万x1万程度の行列での演算でテストを行い、発表を目標とする。