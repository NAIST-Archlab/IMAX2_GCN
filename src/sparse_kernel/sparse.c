#include "../../include/sparse.h"

void spmm(Uint *result, SparseMatrix *sp_matrix, SparseMatrixParams *sp_params, Uint *matrix, int row, int col) {
    Ull CHIP;
    Ull LOOP1, LOOP0;
    Ull INIT1, INIT0;
    Ull AR[64][4];    /* output of EX     in each unit */
    Ull BR[64][4][4]; /* output registers in each unit */
    Ull r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
    Ull r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
    Ull cc0, cc1, cc2, cc3, ex0, ex1;
    Ull cofs, rofs, oofs, k;
    // Ull A_row_size = M, A_col_size = K, B_row_size = K, B_col_size = N;
    Uint blk_iter, A_margin_tmp;
    Ull *A_margin = A_sparse->margin;
    Uint *val_index_set = A_sparse->val_index_set;

    Uint *A_sort_index = A_sparse->sort_index;
    Uint top, blk;
    Ull x, y, z, t;
    Ull x0, y0, z0, t0;
    Ull x1, y1, z1, t1;
    Ull x2, y2, z2, t2;
    Ull x3, y3, z3, t3;
    Ull x4, y4, z4, t4;
    Ull x5, y5, z5, t5;
    Ull x6, y6, z6, t6;
    Ull cofs1;

#undef A_row_size
#undef A_col_size
#undef B_row_size
#undef B_col_size

#undef B_col_blk
#undef NCHIP
#undef W
#undef H
#undef OOFS_SHIFT
#undef A_col_blk
#undef sparse_core1_1
#undef sparse_core2
#undef sparse_core2_1
#undef sparse_core3_1
#undef sparse_core4
#undef sparse_final1

// #define A_row_size 768LL   // 縛りなし
// #define A_col_size 768LL    // 縛りなし　H_padのおかげ
// #define B_row_size 768LL    // 縛りなし
// #define B_col_size 768LL   // B_col_blk*NCHIP縛り

// // #define B_col_blk 16
// #define B_col_blk 8
// /*#define NCHIP 4*/
#define NCHIP 4
    // #define W  4LL
    // #define H  46

    size_t A_row_size = params->A_row_size_param; // 縛りなし
    size_t A_col_size = params->A_col_size_param; // 縛りなし　H_padのおかげ
    size_t B_row_size = params->B_row_size_param; // 縛りなし
    size_t B_col_size = params->B_col_size_param; // B_col_blk*NCHIP縛り
    // #define B_col_blk 16
    size_t B_col_blk = params->B_col_blk_param;
    /*size_t NCHIP 4*/
    // size_t NCHIP = params->NCHIP_param;
    Uint W = params->W_param;
    Uint H = params->H_param;

    size_t B_col_blk_mul_B_row_size = B_col_blk * B_row_size;
    size_t A_row_size_mul_B_col_blk = A_row_size * B_col_blk;
    size_t A_row_size_mul_2_mul_4_mul_2 = A_row_size * 2 * 4 * 2;
    size_t A_row_size_mul_8 = A_row_size * 8;
    size_t cofs_init = (0 - W * 4 * 2 * A_row_size) << 32 | ((0 - W * 4 * 2 * B_row_size) & 0xffffffff);
    size_t rofs_init = (0 - 1 * 8LL) << 32 | ((0 - 1 * 4LL) & 0xffffffff);
    size_t A_row_size_mul_W_mul_4_2_64 = (W * 4 * 2 * A_row_size) << 32 | (W * 4 * 2 * B_row_size);
    typedef struct
    {
        Uint i[8]
    } Ui8;
    Uint *a[H], *a_index[H], *a_debug[H + 1];
    Ui8 *b[NCHIP], *b0[NCHIP], *b1[NCHIP], *b2[NCHIP], *b3[NCHIP];
    Ui8 *c0[NCHIP], *c0_debug[NCHIP];
    Ui8 *c00[NCHIP], *c01[NCHIP], *c02[NCHIP], *c03[NCHIP];

    printf("<<<IMAX>>>\n");
    for (top = 0; top < B_col_size / NCHIP; top += B_col_blk)
    {
        for (blk = 0, blk_iter = 0; blk < A_col_size; blk += H, blk_iter += 1)
        { /* 3重ループ展開の外側対象 */
            if ((A_margin_tmp = A_margin[blk_iter]) == 0)
                break;

            for (CHIP = 0; CHIP < NCHIP; CHIP++)
            {

                b[CHIP] = B + (CHIP * B_col_size / NCHIP + top) * B_row_size;
                b0[CHIP] = b[CHIP] + 0;
                b1[CHIP] = (Uint *)b[CHIP] + B_row_size * 2;
                b2[CHIP] = (Uint *)b[CHIP] + B_row_size * 4;
                b3[CHIP] = (Uint *)b[CHIP] + B_row_size * 6;

                c0[CHIP] = C + (CHIP * B_col_size / NCHIP + top) * A_row_size;
                c00[CHIP] = (Uint *)c0[CHIP] + 0;
                c01[CHIP] = (Uint *)c0[CHIP] + A_row_size * 2;
                c02[CHIP] = (Uint *)c0[CHIP] + A_row_size * 4;
                c03[CHIP] = (Uint *)c0[CHIP] + A_row_size * 6;
            }
            // little edianで格納 big <- <- <- <- small 右から左に流れていく
            // ex 64bit *((Ull*)&offset2)  11000000000000000000000000000000000
            //   32bit *((Uint*)&offset2) -> 0    *((Uint*)&offset2+1) -> 1
            //   8bit  *((char*)&offset2+1) -> 0   *((char*)&offset2+2) -> 0 *((char*)&offset2+3) -> 0 *((char*)&offset2+4) -> 110
            // Ullで扱うときは *((Ull*)&offset2)>>32は*((Uint*)&offset2+1)を取得しているのと等価
            // https://hackaday.com/2020/08/04/dont-let-endianness-flip-you-around/
            // https://www.bogotobogo.com/Embedded/Little_endian_big_endian_htons_htonl.php
            // p *(float*)((Uint*)b[0]+6) 一つ目の値 3
            //>>> p/f  *((Uint*)&BR[2][0][1])
            // $17 = 3
            // >>> p/f  *((Uint*)&BR[2][0][1]+1)
            // $18 = 4
            // UllのA_row_size*2
            // rofs=(0-(Ull)1*8)<<32|((0-(Ull)1*4)&0xffffffff) マスクを使って2パターン使用している
            // 最後のstoreでMSK_D0を他のprogramが使っているので、先例に従った。
            for (k = 0; k < H; k++)
                a[k] = (Uint *)val_index_set + (blk + k) * A_row_size;
            // A_indexはUllなので*2している。 2(32bot*2 load)*4(load unit)*(H/8)=H
            // blk*A_row_size HごとにA_colを進めてる  (Hで何回colを進んだか)*A_row
            // Ullにまとめても取る量自体は同じ              blk/2*A_row_size*2(col/2なので+blk/2  2rowなのでA_row_size*2)
            // Hで割り切れないとき、割り切れない分のa_indexは0が入る  その結果、B_row=0が選ばれるがa[]には0がpadされているのでA[]*B[]=0となり問題ない
            for (k = 0; k < H / 2; k++)
                a_index[k] = (Uint *)val_index_set + blk / 2 * A_row_size * 2 + k * A_row_size * 2 + A_col_size * A_row_size;

                //  A_row_size_mul_2_mul_4_mul_2は二か所で使用する　ブロードキャスト的な？

#define sparse_core1_1(ar, ar_pre, br, br_pre, a_val, index_val, index_MSK)                                                                                     \
    exe(OP_FMA, &AR[ar][0], AR[ar_pre][0], EXP_H3210, BR[br_pre][2][1], EXP_H1010, BR[br_pre][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                      \
    exe(OP_FMA, &AR[ar][1], AR[ar_pre][1], EXP_H3210, BR[br_pre][2][1], EXP_H1010, BR[br_pre][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                      \
    exe(OP_FMA, &AR[ar][2], AR[ar_pre][2], EXP_H3210, BR[br_pre][2][1], EXP_H1010, BR[br_pre][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                      \
    exe(OP_FMA, &AR[ar][3], AR[ar_pre][3], EXP_H3210, BR[br_pre][2][1], EXP_H1010, BR[br_pre][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                      \
    mop(OP_LDR, 3, &BR[br][0][1], (Ull)b0[CHIP], (Ull)index_val, index_MSK, (Ull)b[CHIP], B_col_blk_mul_B_row_size, 0, 0, (Ull)NULL, B_col_blk_mul_B_row_size); \
    mop(OP_LDR, 3, &BR[br][0][0], (Ull)b1[CHIP], (Ull)index_val, index_MSK, (Ull)b[CHIP], B_col_blk_mul_B_row_size, 0, 0, (Ull)NULL, B_col_blk_mul_B_row_size); \
    mop(OP_LDR, 3, &BR[br][1][1], (Ull)b2[CHIP], (Ull)index_val, index_MSK, (Ull)b[CHIP], B_col_blk_mul_B_row_size, 0, 0, (Ull)NULL, B_col_blk_mul_B_row_size); \
    mop(OP_LDR, 3, &BR[br][1][0], (Ull)b3[CHIP], (Ull)index_val, index_MSK, (Ull)b[CHIP], B_col_blk_mul_B_row_size, 0, 0, (Ull)NULL, B_col_blk_mul_B_row_size); \
    mop(OP_LDWR, 1, &BR[br][2][1], (Ull)a[a_val], (Ull)rofs, MSK_W0, (Ull)a[a_val], A_row_size, 0, 0, (Ull)NULL, A_row_size)

#define sparse_core2(ar, ar_pre, br, br_pre, a_index1, a_index2, a_index3, a_index4, a_index_base)                                                                                     \
    exe(OP_FMA, &AR[ar][0], AR[ar_pre][0], EXP_H3210, BR[br_pre][2][1], EXP_H1010, BR[br_pre][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                             \
    exe(OP_FMA, &AR[ar][1], AR[ar_pre][1], EXP_H3210, BR[br_pre][2][1], EXP_H1010, BR[br_pre][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                             \
    exe(OP_FMA, &AR[ar][2], AR[ar_pre][2], EXP_H3210, BR[br_pre][2][1], EXP_H1010, BR[br_pre][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                             \
    exe(OP_FMA, &AR[ar][3], AR[ar_pre][3], EXP_H3210, BR[br_pre][2][1], EXP_H1010, BR[br_pre][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                             \
    mop(OP_LDR, 3, &BR[br][1][1], (Ull)a_index[a_index1], (Ull)rofs, MSK_W1, (Ull)a_index[a_index_base], A_row_size_mul_2_mul_4_mul_2, 0, 0, (Ull)NULL, A_row_size_mul_2_mul_4_mul_2); \
    mop(OP_LDR, 3, &BR[br][1][0], (Ull)a_index[a_index2], (Ull)rofs, MSK_W1, (Ull)a_index[a_index_base], A_row_size_mul_2_mul_4_mul_2, 0, 0, (Ull)NULL, A_row_size_mul_2_mul_4_mul_2); \
    mop(OP_LDR, 3, &BR[br][2][1], (Ull)a_index[a_index3], (Ull)rofs, MSK_W1, (Ull)a_index[a_index_base], A_row_size_mul_2_mul_4_mul_2, 0, 0, (Ull)NULL, A_row_size_mul_2_mul_4_mul_2); \
    mop(OP_LDR, 3, &BR[br][2][0], (Ull)a_index[a_index4], (Ull)rofs, MSK_W1, (Ull)a_index[a_index_base], A_row_size_mul_2_mul_4_mul_2, 0, 0, (Ull)NULL, A_row_size_mul_2_mul_4_mul_2)

#define sparse_core2_1(ar, ar_pre, br, br_pre, a_index1, a_index2, a_index3, a_index_base)                                                                                             \
    exe(OP_FMA, &AR[ar][0], AR[ar_pre][0], EXP_H3210, BR[br_pre][2][1], EXP_H1010, BR[br_pre][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                             \
    exe(OP_FMA, &AR[ar][1], AR[ar_pre][1], EXP_H3210, BR[br_pre][2][1], EXP_H1010, BR[br_pre][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                             \
    exe(OP_FMA, &AR[ar][2], AR[ar_pre][2], EXP_H3210, BR[br_pre][2][1], EXP_H1010, BR[br_pre][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                             \
    exe(OP_FMA, &AR[ar][3], AR[ar_pre][3], EXP_H3210, BR[br_pre][2][1], EXP_H1010, BR[br_pre][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                             \
    mop(OP_LDR, 3, &BR[br][1][1], (Ull)a_index[a_index1], (Ull)rofs, MSK_W1, (Ull)a_index[a_index_base], A_row_size_mul_2_mul_4_mul_2, 0, 0, (Ull)NULL, A_row_size_mul_2_mul_4_mul_2); \
    mop(OP_LDR, 3, &BR[br][1][0], (Ull)a_index[a_index2], (Ull)rofs, MSK_W1, (Ull)a_index[a_index_base], A_row_size_mul_2_mul_4_mul_2, 0, 0, (Ull)NULL, A_row_size_mul_2_mul_4_mul_2); \
    mop(OP_LDR, 3, &BR[br][2][1], (Ull)a_index[a_index3], (Ull)rofs, MSK_W1, (Ull)a_index[a_index_base], A_row_size_mul_2_mul_4_mul_2, 0, 0, (Ull)NULL, A_row_size_mul_2_mul_4_mul_2); \
    mop(OP_LDR, 3, &BR[br][2][0], (Ull)a_index[a_index3], (Ull)rofs, MSK_W1, (Ull)a_index[a_index_base], A_row_size_mul_2_mul_4_mul_2, 0, 0, (Ull)NULL, A_row_size_mul_2_mul_4_mul_2)

#define sparse_core3_1(ar, ar_pre, br, br_pre, a_val, index_val1, index_val2, index_val3, index_val4, index_val, index_MSK)                                \
    exe(OP_ADD, &index_val1, BR[br_pre][1][1], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                     \
    exe(OP_ADD, &index_val2, BR[br_pre][1][0], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                     \
    exe(OP_ADD, &index_val3, BR[br_pre][2][1], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                     \
    exe(OP_ADD, &index_val4, BR[br_pre][2][0], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                     \
    mop(OP_LDR, 3, &BR[br][0][1], (Ull)b0[CHIP], index_val, index_MSK, (Ull)b[CHIP], B_col_blk_mul_B_row_size, 0, 0, (Ull)NULL, B_col_blk_mul_B_row_size); \
    mop(OP_LDR, 3, &BR[br][0][0], (Ull)b1[CHIP], index_val, index_MSK, (Ull)b[CHIP], B_col_blk_mul_B_row_size, 0, 0, (Ull)NULL, B_col_blk_mul_B_row_size); \
    mop(OP_LDR, 3, &BR[br][1][1], (Ull)b2[CHIP], index_val, index_MSK, (Ull)b[CHIP], B_col_blk_mul_B_row_size, 0, 0, (Ull)NULL, B_col_blk_mul_B_row_size); \
    mop(OP_LDR, 3, &BR[br][1][0], (Ull)b3[CHIP], index_val, index_MSK, (Ull)b[CHIP], B_col_blk_mul_B_row_size, 0, 0, (Ull)NULL, B_col_blk_mul_B_row_size); \
    mop(OP_LDWR, 1, &BR[br][2][1], (Ull)a[a_val], (Ull)rofs, MSK_W0, (Ull)a[a_val], A_row_size, 0, 0, (Ull)NULL, A_row_size)

#define sparse_core4(ar, ar_pre, br, br_pre)                                                                                               \
    exe(OP_FMA, &AR[ar][0], AR[ar_pre][0], EXP_H3210, BR[br_pre][2][1], EXP_H1010, BR[br_pre][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); \
    exe(OP_FMA, &AR[ar][1], AR[ar_pre][1], EXP_H3210, BR[br_pre][2][1], EXP_H1010, BR[br_pre][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); \
    exe(OP_FMA, &AR[ar][2], AR[ar_pre][2], EXP_H3210, BR[br_pre][2][1], EXP_H1010, BR[br_pre][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); \
    exe(OP_FMA, &AR[ar][3], AR[ar_pre][3], EXP_H3210, BR[br_pre][2][1], EXP_H1010, BR[br_pre][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); \
    mop(OP_LDWR, 1, &BR[br][0][1], (Ull)A_sort_index, (Ull)rofs, MSK_W0, (Ull)A_sort_index, A_row_size, 0, 0, (Ull)NULL, A_row_size)

#define sparse_final1(ar, ar_pre, br, br_pre, index_val)                                                                                                       \
    mop(OP_LDR, 3, &BR[br][0][1], (Ull)c00[CHIP], (Ull)index_val, MSK_W0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 1, (Ull)NULL, A_row_size_mul_B_col_blk); \
    mop(OP_LDR, 3, &BR[br][1][1], (Ull)c01[CHIP], (Ull)index_val, MSK_W0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 1, (Ull)NULL, A_row_size_mul_B_col_blk); \
    mop(OP_LDR, 3, &BR[br][2][1], (Ull)c02[CHIP], (Ull)index_val, MSK_W0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 1, (Ull)NULL, A_row_size_mul_B_col_blk); \
    mop(OP_LDR, 3, &BR[br][3][1], (Ull)c03[CHIP], (Ull)index_val, MSK_W0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 1, (Ull)NULL, A_row_size_mul_B_col_blk); \
    exe(OP_FAD, &AR[ar][0], AR[ar_pre][0], EXP_H3210, BR[br][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                      \
    exe(OP_FAD, &AR[ar][1], AR[ar_pre][1], EXP_H3210, BR[br][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                      \
    exe(OP_FAD, &AR[ar][2], AR[ar_pre][2], EXP_H3210, BR[br][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                      \
    exe(OP_FAD, &AR[ar][3], AR[ar_pre][3], EXP_H3210, BR[br][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                      \
    mop(OP_STR, 3, &AR[ar][0], (Ull)index_val, (Ull)c00[CHIP], MSK_D0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 1, (Ull)NULL, A_row_size_mul_B_col_blk);    \
    mop(OP_STR, 3, &AR[ar][1], (Ull)index_val, (Ull)c01[CHIP], MSK_D0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 1, (Ull)NULL, A_row_size_mul_B_col_blk);    \
    mop(OP_STR, 3, &AR[ar][2], (Ull)index_val, (Ull)c02[CHIP], MSK_D0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 1, (Ull)NULL, A_row_size_mul_B_col_blk);    \
    mop(OP_STR, 3, &AR[ar][3], (Ull)index_val, (Ull)c03[CHIP], MSK_D0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 1, (Ull)NULL, A_row_size_mul_B_col_blk)

            // EMAX5A begin mm1 mapdist=0
            /*3*/ for (CHIP = 0; CHIP < NCHIP; CHIP++)
            {   /* will be parallelized by multi-chip (M/#chip) */
                // LOOP1--はLOOP1==1で終了するので、LOOP1はrow+1がよい
                /*1*/ for (INIT1 = 1, LOOP1 = B_col_blk / (W * 2), cofs = cofs_init; LOOP1--; INIT1 = 0)
                { /* stage#0 */ /* mapped to FOR() on BR[63][0][0] */
                    /*2*/ for (INIT0 = 1, LOOP0 = A_margin_tmp, rofs = rofs_init; LOOP0--; INIT0 = 0)
                    { /* stage#0 */                                                                                                                                                           /* mapped to FOR() on BR[63][1][0] */
                        exe(OP_ADD, &cofs, cofs, EXP_H3210, INIT0 ? A_row_size_mul_W_mul_4_2_64 : 0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                    /* stage#0 */
                        exe(OP_ADD, &rofs, INIT0 ? rofs : rofs, EXP_H3210, (1 * 8LL) << 32 | ((1 * 4LL) & 0xffffffff), EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL); /* stage#0 */
                        exe(OP_ADD, &cofs1, cofs, EXP_H1010, 0, EXP_H3210, 0, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0);                                                                             /* stage#1 */
                        exe(OP_ADD, &oofs, cofs, EXP_H3232, 0, EXP_H3210, 0, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0);                                                                              /* stage#1 */
                        mop(OP_LDR, 3, &BR[1][1][1], (Ull)a_index[0], (Ull)rofs, MSK_W1, (Ull)a_index[0], A_row_size_mul_2_mul_4_mul_2, 0, 0, (Ull)NULL, A_row_size_mul_2_mul_4_mul_2);       /* stage#1 */
                        mop(OP_LDR, 3, &BR[1][1][0], (Ull)a_index[1], (Ull)rofs, MSK_W1, (Ull)a_index[0], A_row_size_mul_2_mul_4_mul_2, 0, 0, (Ull)NULL, A_row_size_mul_2_mul_4_mul_2);       /* stage#1 */
                        mop(OP_LDR, 3, &BR[1][2][1], (Ull)a_index[2], (Ull)rofs, MSK_W1, (Ull)a_index[0], A_row_size_mul_2_mul_4_mul_2, 0, 0, (Ull)NULL, A_row_size_mul_2_mul_4_mul_2);       /* stage#1 */
                        mop(OP_LDR, 3, &BR[1][2][0], (Ull)a_index[3], (Ull)rofs, MSK_W1, (Ull)a_index[0], A_row_size_mul_2_mul_4_mul_2, 0, 0, (Ull)NULL, A_row_size_mul_2_mul_4_mul_2);       /* stage#1 */

                        exe(OP_ADD, &x, BR[1][1][1], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                       /* stage#2 */
                        exe(OP_ADD, &y, BR[1][1][0], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                       /* stage#2 */
                        exe(OP_ADD, &z, BR[1][2][1], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                       /* stage#2 */
                        exe(OP_ADD, &t, BR[1][2][0], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                       /* stage#2 */
                        mop(OP_LDR, 3, &BR[3][0][1], (Ull)b0[CHIP], x, MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size, 0, 0, (Ull)NULL, B_col_blk_mul_B_row_size); /* stage#2 */
                        mop(OP_LDR, 3, &BR[3][0][0], (Ull)b1[CHIP], x, MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size, 0, 0, (Ull)NULL, B_col_blk_mul_B_row_size); /* stage#2 */
                        mop(OP_LDR, 3, &BR[3][1][1], (Ull)b2[CHIP], x, MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size, 0, 0, (Ull)NULL, B_col_blk_mul_B_row_size); /* stage#2 */
                        mop(OP_LDR, 3, &BR[3][1][0], (Ull)b3[CHIP], x, MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size, 0, 0, (Ull)NULL, B_col_blk_mul_B_row_size); /* stage#2 2KB */
                        mop(OP_LDWR, 1, &BR[3][2][1], (Ull)a[0], (Ull)rofs, MSK_W0, (Ull)a[0], A_row_size, 0, 0, (Ull)NULL, A_row_size);                           /* stage#2 16KB */

                        exe(OP_FML, &AR[4][0], BR[3][0][1], EXP_H3210, BR[3][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                        exe(OP_FML, &AR[4][1], BR[3][0][0], EXP_H3210, BR[3][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                        exe(OP_FML, &AR[4][2], BR[3][1][1], EXP_H3210, BR[3][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                        exe(OP_FML, &AR[4][3], BR[3][1][0], EXP_H3210, BR[3][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
                        mop(OP_LDR, 3, &BR[4][0][1], (Ull)b0[CHIP], (Ull)x, MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size, 0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);
                        mop(OP_LDR, 3, &BR[4][0][0], (Ull)b1[CHIP], (Ull)x, MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size, 0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);
                        mop(OP_LDR, 3, &BR[4][1][1], (Ull)b2[CHIP], (Ull)x, MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size, 0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);
                        mop(OP_LDR, 3, &BR[4][1][0], (Ull)b3[CHIP], (Ull)x, MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size, 0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);
                        mop(OP_LDWR, 1, &BR[4][2][1], (Ull)a[1], (Ull)rofs, MSK_W0, (Ull)a[1], A_row_size, 0, 0, (Ull)NULL, A_row_size);

                        sparse_core1_1(5, 4, 5, 4, 2, y, MSK_W0);
                        sparse_core1_1(6, 5, 6, 5, 3, y, MSK_W1);
                        sparse_core1_1(7, 6, 7, 6, 4, z, MSK_W0);
                        sparse_core1_1(8, 7, 8, 7, 5, z, MSK_W1);
                        sparse_core1_1(9, 8, 9, 8, 6, t, MSK_W0);
                        sparse_core1_1(10, 9, 10, 9, 7, t, MSK_W1);

                        sparse_core2(11, 10, 11, 10, 4, 5, 6, 7, 0);
                        sparse_core3_1(12, 11, 13, 11, 8, x, y, z, t, x, MSK_W0);

                        sparse_core1_1(14, 11, 14, 13, 9, x, MSK_W1);

                        sparse_core1_1(15, 14, 15, 14, 10, y, MSK_W0);
                        sparse_core1_1(16, 15, 16, 15, 11, y, MSK_W1);
                        sparse_core1_1(17, 16, 17, 16, 12, z, MSK_W0);
                        sparse_core1_1(18, 17, 18, 17, 13, z, MSK_W1);
                        sparse_core1_1(19, 18, 19, 18, 14, t, MSK_W0);
                        sparse_core1_1(20, 19, 20, 19, 15, t, MSK_W1);

                        sparse_core2(21, 20, 21, 20, 8, 9, 10, 11, 8);
                        sparse_core3_1(22, 21, 23, 21, 16, x, y, z, t, x, MSK_W0);

                        sparse_core1_1(24, 21, 24, 23, 17, x, MSK_W1);

                        sparse_core1_1(25, 24, 25, 24, 18, y, MSK_W0);
                        sparse_core1_1(26, 25, 26, 25, 19, y, MSK_W1);
                        sparse_core1_1(27, 26, 27, 26, 20, z, MSK_W0);
                        sparse_core1_1(28, 27, 28, 27, 21, z, MSK_W1);
                        sparse_core1_1(29, 28, 29, 28, 22, t, MSK_W0);
                        sparse_core1_1(30, 29, 30, 29, 23, t, MSK_W1);

                        sparse_core2(31, 30, 31, 30, 12, 13, 14, 15, 8);
                        sparse_core3_1(32, 31, 33, 31, 24, x, y, z, t, x, MSK_W0);

                        sparse_core1_1(34, 31, 34, 33, 25, x, MSK_W1);

                        sparse_core1_1(35, 34, 35, 34, 26, y, MSK_W0);
                        sparse_core1_1(36, 35, 36, 35, 27, y, MSK_W1);
                        sparse_core1_1(37, 36, 37, 36, 28, z, MSK_W0);
                        sparse_core1_1(38, 37, 38, 37, 29, z, MSK_W1);
                        sparse_core1_1(39, 38, 39, 38, 30, t, MSK_W0);
                        sparse_core1_1(40, 39, 40, 39, 31, t, MSK_W1);

                        sparse_core2(41, 40, 41, 40, 16, 17, 18, 19, 16);
                        sparse_core3_1(42, 41, 43, 41, 32, x, y, z, t, x, MSK_W0);

                        sparse_core1_1(44, 41, 44, 43, 33, x, MSK_W1);

                        sparse_core1_1(45, 44, 45, 44, 34, y, MSK_W0);
                        sparse_core1_1(46, 45, 46, 45, 35, y, MSK_W1);
                        sparse_core1_1(47, 46, 47, 46, 36, z, MSK_W0);
                        sparse_core1_1(48, 47, 48, 47, 37, z, MSK_W1);
                        sparse_core1_1(49, 48, 49, 48, 38, t, MSK_W0);
                        sparse_core1_1(50, 49, 50, 49, 39, t, MSK_W1);

                        sparse_core2_1(51, 50, 51, 50, 20, 21, 22, 16);
                        sparse_core3_1(52, 51, 53, 51, 40, x, y, z, t, x, MSK_W0);

                        sparse_core1_1(54, 51, 54, 53, 41, x, MSK_W1);

                        sparse_core1_1(55, 54, 55, 54, 42, y, MSK_W0);
                        sparse_core1_1(56, 55, 56, 55, 43, y, MSK_W1);
                        sparse_core1_1(57, 56, 57, 56, 44, z, MSK_W0);
                        sparse_core1_1(58, 57, 58, 57, 45, z, MSK_W1);
                        sparse_core4(59, 58, 59, 58);
                        exe(OP_ADD, &x, BR[59][0][1], EXP_H3210, oofs, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffff, OP_NOP, 0LL);

                        sparse_final1(62, 59, 62, 59, x);
                    }
                }
            }
            // EMAX5A end
        }
    }
    // EMAX5A drain_dirty_lmm
}