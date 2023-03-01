#include "../include/sparse.h"
#include "../include/emax6.h"
#include "../include/emax6lib.c"
#ifdef USE_IMAX2
#include <stdio.h>

void spmm(float *result, SparseMatrix *sp_matrix, SparseMatrixParams *sp_params, float *matrix, int mm_col) {
    Ull CHIP; Ull LOOP1, LOOP0; Ull INIT1, INIT0;
    Ull top, blk, blk_iter;
    Ull AR[64][4]; /* output of EX in each unit */
    Ull BR[64][4][4]; /* output registers in each unit */
    Ull r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
    Ull r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
    Ull cc0, cc1, cc2, cc3, ex0, ex1; Ull cofs, rofs, oofs, k, row_p, cofs1;

    #define NCHIP 4
    Ull *A_margin;
    Ull A_margin_tmp;
    Uint W = 8;
    Uint H = 46;

    size_t A_row_size = sp_matrix->row_size;
    size_t A_col_size = sp_matrix->col_size;
    size_t B_row_size = sp_matrix->col_size;
    size_t B_col_size = mm_col;
    size_t B_col_blk;
    size_t cofs_init = (0-W*4*2*A_row_size)<<32|((0-W*4*1*B_row_size)&0xffffffff);
    size_t rofs_init = (0-1*8LL)<<32|((0-1*4LL)*0xffffffff);

    size_t B_col_blk_mul_B_row_size = B_col_blk*B_row_size;
    size_t A_row_size_mul_B_col_blk = A_row_size*B_col_blk;
    size_t A_row_size_mul_2_4_2 = A_row_size*2*4*2;
    size_t A_row_size_mul_8 = A_row_size*8;
    size_t A_row_size_mul_W_4_2 = (W*4*2*A_row_size)<<32|(W*4*2*B_row_size);

    typedef struct {Uint i[8]} Ui8;
    Uint *a_col_index[H], *a[H];
    Ui8 *b[NCHIP], *b0[NCHIP], *b1[NCHIP], *b2[NCHIP], *b3[NCHIP];
    Ui8 *c0[NCHIP];
    Ui8 *c00[NCHIP], *c01[NCHIP], *c02[NCHIP], *c03[NCHIP];

    for (top=0; top < B_col_size/NCHIP; top+=B_col_blk) {
        for (blk=0, blk_iter=0; blk < A_col_size; blk+=H,blk_iter+=1) {
            if((A_margin_tmp=A_margin[blk_iter])==0) break;

            for (CHIP=0; CHIP<NCHIP; CHIP++) {
                b[CHIP] = matrix+(CHIP*B_col_size/NCHIP + top)*B_row_size; 
                b0[CHIP] = (Uint*)b[CHIP];
                b1[CHIP] = (Uint*)b[CHIP] + B_row_size*2;
                b2[CHIP] = (Uint*)b[CHIP] + B_row_size*4;
                b3[CHIP] = (Uint*)b[CHIP] + B_row_size*6;

                c0[CHIP] = result + (CHIP*B_col_size/NCHIP+top)*A_row_size;
                c00[CHIP]= (Uint*)c0[CHIP] + 0;
                c01[CHIP]= (Uint*)c0[CHIP] + A_row_size*2;
                c02[CHIP]= (Uint*)c0[CHIP] + A_row_size*4;
                c03[CHIP]= (Uint*)c0[CHIP] + A_row_size*6;
            }

            for (k=0; k < H; k++) a[k] = (Uint*)sp_matrix->val + (blk+k)*A_row_size;
            for (k=0; k < H/2; k++) a_col_index[k] = (Uint*)sp_matrix->col_p + blk/2*A_row_size*2 + k*A_row_size*2 + A_col_size*A_row_size;

            #define spmm_core1(r, rm1, a_row, offset, msk) \
                        exe(OP_FMA, &AR[r][0], AR[rm1][0], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                        exe(OP_FMA, &AR[r][1], AR[rm1][1], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                        exe(OP_FMA, &AR[r][2], AR[rm1][2], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                        exe(OP_FMA, &AR[r][3], AR[rm1][3], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                        mop(OP_LDR,  3, &BR[r][0][1], (Ull)b0[CHIP], (Ull)offset,   msk, (Ull)b[0], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                        mop(OP_LDR,  3, &BR[r][0][0], (Ull)b1[CHIP], (Ull)offset,   msk, (Ull)b[0], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                        mop(OP_LDR,  3, &BR[r][1][1], (Ull)b2[CHIP], (Ull)offset,   msk, (Ull)b[0], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                        mop(OP_LDR,  3, &BR[r][1][0], (Ull)b3[CHIP], (Ull)offset,   msk, (Ull)b[0], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                        mop(OP_LDWR, 1, &BR[r][2][1], (Ull)a[a_row],  (Ull)rofs, MSK_W0, (Ull)a[a_row], A_row_size, 0, 0, (Ull)NULL, A_row_size)

            #define spmm_core1_end(r, rm1, idx0, idx1, idx2, idx3, idx_base) \
                        exe(OP_FMA, &AR[r][0], AR[rm1][0], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                        exe(OP_FMA, &AR[r][1], AR[rm1][1], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                        exe(OP_FMA, &AR[r][2], AR[rm1][2], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                        exe(OP_FMA, &AR[r][3], AR[rm1][3], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                        mop(OP_LDR,  3, &BR[r][1][1], (Ull)a_col_index[idx0], (Ull)rofs,   MSK_W1, (Ull)a_col_index[idx_base], A_row_size_mul_2_4_2,    0, 0, (Ull)NULL, A_row_size_mul_2_4_2);\
                        mop(OP_LDR,  3, &BR[r][1][0], (Ull)a_col_index[idx1], (Ull)rofs,   MSK_W1, (Ull)a_col_index[idx_base], A_row_size_mul_2_4_2,    0, 0, (Ull)NULL, A_row_size_mul_2_4_2);\
                        mop(OP_LDR,  3, &BR[r][2][1], (Ull)a_col_index[idx2], (Ull)rofs,   MSK_W1, (Ull)a_col_index[idx_base], A_row_size_mul_2_4_2,    0, 0, (Ull)NULL, A_row_size_mul_2_4_2);\
                        mop(OP_LDR,  3, &BR[r][2][0], (Ull)a_col_index[idx3], (Ull)rofs,   MSK_W1, (Ull)a_col_index[idx_base], A_row_size_mul_2_4_2,    0, 0, (Ull)NULL, A_row_size_mul_2_4_2)

            #define spmm_core1_start(rp2, rp1, r, rm1, a_rowp1, a_row) \
                        exe(OP_ADD, &r0, BR[rm1][1][1], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                        exe(OP_ADD, &r1, BR[rm1][1][0], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                        exe(OP_ADD, &r2, BR[rm1][2][1], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                        exe(OP_ADD, &r3, BR[rm1][2][0], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                        mop(OP_LDR,  3, &BR[rp1][0][1], (Ull)b0[CHIP], (Ull)r0,   MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                        mop(OP_LDR,  3, &BR[rp1][0][0], (Ull)b1[CHIP], (Ull)r0,   MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                        mop(OP_LDR,  3, &BR[rp1][1][1], (Ull)b2[CHIP], (Ull)r0,   MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                        mop(OP_LDR,  3, &BR[rp1][1][0], (Ull)b3[CHIP], (Ull)r0,   MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                        mop(OP_LDWR, 1, &BR[rp1][2][1], (Ull)a[a_row],  (Ull)rofs, MSK_W0, (Ull)a[a_row], A_row_size, 0, 0, (Ull)NULL, A_row_size);\
                        exe(OP_FMA, &AR[rp2][0], AR[rm1][0], EXP_H3210, BR[rp1][2][1], EXP_H1010, BR[r][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                        exe(OP_FMA, &AR[rp2][1], AR[rm1][1], EXP_H3210, BR[rp1][2][1], EXP_H1010, BR[r][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                        exe(OP_FMA, &AR[rp2][2], AR[rm1][2], EXP_H3210, BR[rp1][2][1], EXP_H1010, BR[r][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                        exe(OP_FMA, &AR[rp2][3], AR[rm1][3], EXP_H3210, BR[rp1][2][1], EXP_H1010, BR[r][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                        mop(OP_LDR,  3, &BR[rp2][0][1], (Ull)b0[CHIP], (Ull)r0,   MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                        mop(OP_LDR,  3, &BR[rp2][0][0], (Ull)b1[CHIP], (Ull)r0,   MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                        mop(OP_LDR,  3, &BR[rp2][1][1], (Ull)b2[CHIP], (Ull)r0,   MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                        mop(OP_LDR,  3, &BR[rp2][1][0], (Ull)b3[CHIP], (Ull)r0,   MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                        mop(OP_LDWR, 1, &BR[rp2][2][1], (Ull)a[a_rowp1],  (Ull)rofs, MSK_W0, (Ull)a[a_rowp1], A_row_size, 0, 0, (Ull)NULL, A_row_size)

            #define spmm_final(r, rm1) \
                        mop(OP_LDR, 3, &BR[r][0][1], (Ull)c00[CHIP], (Ull)r0, MSK_W0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 1, (Ull)NULL, A_row_size_mul_B_col_blk);\
                        mop(OP_LDR, 3, &BR[r][1][1], (Ull)c01[CHIP], (Ull)r0, MSK_W0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 1, (Ull)NULL, A_row_size_mul_B_col_blk);\
                        mop(OP_LDR, 3, &BR[r][2][1], (Ull)c02[CHIP], (Ull)r0, MSK_W0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 1, (Ull)NULL, A_row_size_mul_B_col_blk);\
                        mop(OP_LDR, 3, &BR[r][3][1], (Ull)c03[CHIP], (Ull)r0, MSK_W0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 1, (Ull)NULL, A_row_size_mul_B_col_blk);\
                        exe(OP_FAD, &AR[r][0], AR[rm1][0], EXP_H3210, BR[r][0][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                        exe(OP_FAD, &AR[r][1], AR[rm1][1], EXP_H3210, BR[r][1][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                        exe(OP_FAD, &AR[r][2], AR[rm1][2], EXP_H3210, BR[r][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                        exe(OP_FAD, &AR[r][3], AR[rm1][3], EXP_H3210, BR[r][3][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                        mop(OP_STR, 3, &AR[r][0], (Ull)r0, (Ull)c00[CHIP], MSK_D0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 1, (Ull)NULL, A_row_size_mul_B_col_blk);\
                        mop(OP_STR, 3, &AR[r][1], (Ull)r0, (Ull)c01[CHIP], MSK_D0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 1, (Ull)NULL, A_row_size_mul_B_col_blk);\
                        mop(OP_STR, 3, &AR[r][2], (Ull)r0, (Ull)c02[CHIP], MSK_D0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 1, (Ull)NULL, A_row_size_mul_B_col_blk);\
                        mop(OP_STR, 3, &AR[r][3], (Ull)r0, (Ull)c03[CHIP], MSK_D0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 1, (Ull)NULL, A_row_size_mul_B_col_blk)

        //EMAX5A begin spmm1 mapdist=0
            for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
                for (INIT1=1,LOOP1=B_col_blk/(W*2),cofs=cofs_init; LOOP1--; INIT1=0) {
                    for (INIT0=1,LOOP0=A_margin_tmp,rofs=rofs_init; LOOP0--; INIT0=0) {
                        exe(OP_ADD, &cofs,  cofs,            EXP_H3210, INIT0?A_row_size_mul_W_4_2:0,     EXP_H3210, 0LL,    EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                        exe(OP_ADD, &rofs,  INIT0?rofs:rofs, EXP_H3210, (1*8LL)<<32|((1*4LL)&0xffffffff), EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0xffffffffffffffffLL, OP_NOP, 0LL);
                        exe(OP_ADD, &cofs1, cofs,            EXP_H1010, 0,                                EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL);
                        exe(OP_ADD, &oofs,  cofs,            EXP_H3232, 0,                                EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL);
                        mop(OP_LDR,  3, &BR[1][1][1], (Ull)a_col_index[0], (Ull)rofs, MSK_W1, (Ull)a_col_index[0], A_row_size_mul_2_4_2, 0, 0, (Ull)NULL, A_row_size_mul_2_4_2);
                        mop(OP_LDR,  3, &BR[1][1][0], (Ull)a_col_index[1], (Ull)rofs, MSK_W1, (Ull)a_col_index[0], A_row_size_mul_2_4_2, 0, 0, (Ull)NULL, A_row_size_mul_2_4_2);
                        mop(OP_LDR,  3, &BR[1][2][1], (Ull)a_col_index[2], (Ull)rofs, MSK_W1, (Ull)a_col_index[0], A_row_size_mul_2_4_2, 0, 0, (Ull)NULL, A_row_size_mul_2_4_2);
                        mop(OP_LDR,  3, &BR[1][2][0], (Ull)a_col_index[3], (Ull)rofs, MSK_W1, (Ull)a_col_index[0], A_row_size_mul_2_4_2, 0, 0, (Ull)NULL, A_row_size_mul_2_4_2);

                        exe(OP_ADD, &r0, BR[1][1][1], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                        exe(OP_ADD, &r1, BR[1][1][0], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                        exe(OP_ADD, &r2, BR[1][2][1], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                        exe(OP_ADD, &r3, BR[1][2][0], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                        mop(OP_LDR,  3, &BR[3][0][1], (Ull)b0[CHIP], (Ull)r0,   MSK_W0, (Ull)b[0], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);
                        mop(OP_LDR,  3, &BR[3][0][0], (Ull)b1[CHIP], (Ull)r0,   MSK_W0, (Ull)b[0], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);
                        mop(OP_LDR,  3, &BR[3][1][1], (Ull)b2[CHIP], (Ull)r0,   MSK_W0, (Ull)b[0], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);
                        mop(OP_LDR,  3, &BR[3][1][0], (Ull)b3[CHIP], (Ull)r0,   MSK_W0, (Ull)b[0], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);
                        mop(OP_LDWR, 1, &BR[3][2][1], (Ull)a[0],  (Ull)rofs, MSK_W0, (Ull)a[0], A_row_size, 0, 0, (Ull)NULL, A_row_size);

                        exe(OP_FML, &AR[4][0], BR[3][0][1], EXP_H3210, BR[3][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                        exe(OP_FML, &AR[4][1], BR[3][0][0], EXP_H3210, BR[3][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                        exe(OP_FML, &AR[4][2], BR[3][1][1], EXP_H3210, BR[3][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                        exe(OP_FML, &AR[4][3], BR[3][1][0], EXP_H3210, BR[3][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                        mop(OP_LDR,  3, &BR[4][0][1], (Ull)b0[CHIP], (Ull)r0,   MSK_W1, (Ull)b[0], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);
                        mop(OP_LDR,  3, &BR[4][0][0], (Ull)b1[CHIP], (Ull)r0,   MSK_W1, (Ull)b[0], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);
                        mop(OP_LDR,  3, &BR[4][1][1], (Ull)b2[CHIP], (Ull)r0,   MSK_W1, (Ull)b[0], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);
                        mop(OP_LDR,  3, &BR[4][1][0], (Ull)b3[CHIP], (Ull)r0,   MSK_W1, (Ull)b[0], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);
                        mop(OP_LDWR, 1, &BR[4][2][1], (Ull)a[1],  (Ull)rofs, MSK_W0, (Ull)a[1], A_row_size, 0, 0, (Ull)NULL, A_row_size);

                        spmm_core1(5,  4, 2, r1, MSK_W0);
                        spmm_core1(6,  5, 3, r1, MSK_W1);
                        spmm_core1(7,  6, 4, r2, MSK_W0);
                        spmm_core1(8,  7, 5, r2, MSK_W1);
                        spmm_core1(9,  8, 6, r3, MSK_W0);
                        spmm_core1(10, 9, 7, r3, MSK_W1);
                        spmm_core1_end(11, 10, 4, 5, 6, 7, 0);

                        spmm_core1_start(14, 13, 12, 11, 9, 8);
                        spmm_core1(15, 14, 10, r1, MSK_W0);
                        spmm_core1(16, 15, 11, r1, MSK_W1);
                        spmm_core1(17, 16, 12, r2, MSK_W0);
                        spmm_core1(18, 17, 13, r2, MSK_W1);
                        spmm_core1(19, 18, 14, r3, MSK_W0);
                        spmm_core1(20, 19, 15, r3, MSK_W1);
                        spmm_core1_end(21, 20, 8, 9, 10, 11, 8);

                        spmm_core1_start(24, 23, 22, 21, 17, 16);
                        spmm_core1(25, 24, 18, r1, MSK_W0);
                        spmm_core1(26, 25, 19, r1, MSK_W1);
                        spmm_core1(27, 26, 20, r2, MSK_W0);
                        spmm_core1(28, 27, 21, r2, MSK_W1);
                        spmm_core1(29, 28, 22, r3, MSK_W0);
                        spmm_core1(30, 29, 23, r3, MSK_W1);
                        spmm_core1_end(31, 30, 12, 13, 14, 15, 8);

                        spmm_core1_start(34, 33, 32, 31, 25, 24);
                        spmm_core1(35, 34, 26, r1, MSK_W0);
                        spmm_core1(36, 35, 27, r1, MSK_W1);
                        spmm_core1(37, 36, 28, r2, MSK_W0);
                        spmm_core1(38, 37, 29, r2, MSK_W1);
                        spmm_core1(39, 38, 30, r3, MSK_W0);
                        spmm_core1(40, 29, 31, r3, MSK_W1);
                        spmm_core1_end(41, 40, 16, 17, 18, 19, 16);

                        spmm_core1_start(44, 43, 42, 41, 33, 32);
                        spmm_core1(45, 44, 34, r1, MSK_W0);
                        spmm_core1(46, 45, 35, r1, MSK_W1);
                        spmm_core1(47, 46, 36, r2, MSK_W0);
                        spmm_core1(48, 47, 37, r2, MSK_W1);
                        spmm_core1(49, 48, 38, r3, MSK_W0);
                        spmm_core1(50, 49, 39, r3, MSK_W1);
                        spmm_core1_end(51, 50, 20, 21, 22, 23, 16);

                        spmm_core1_start(54, 53, 52, 51, 41, 40);
                        spmm_core1(55, 54, 42, r1, MSK_W0);
                        spmm_core1(56, 55, 43, r1, MSK_W1);
                        spmm_core1(57, 56, 44, r2, MSK_W0);
                        spmm_core1(58, 57, 45, r2, MSK_W1);

                        exe(OP_FMA, &AR[59][0], AR[58][0], EXP_H3210, BR[58][2][1], EXP_H1010, BR[58][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                        exe(OP_FMA, &AR[59][1], AR[58][1], EXP_H3210, BR[58][2][1], EXP_H1010, BR[58][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                        exe(OP_FMA, &AR[59][2], AR[58][2], EXP_H3210, BR[58][2][1], EXP_H1010, BR[58][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                        exe(OP_FMA, &AR[59][3], AR[58][3], EXP_H3210, BR[58][2][1], EXP_H1010, BR[58][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                        mop(OP_LDWR, 1, &BR[59][0][1], (Ull)a_col_index, (Ull)rofs, MSK_W1, (Ull)a_col_index, B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);

                        exe(OP_ADD, &r0, BR[59][0][1], EXP_H3210, oofs, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xfffffffLL, OP_NOP, 0LL);
                        spmm_final(62, 59);
                    }
                }
            }
        //EMAX5A end
        }
        //EMAX5A drain_dirty_lmm
    }
}

void mm(float *result, float *a, float *b, int row_a, int col_a, int col_b) {
    for (int i = 0; i < row_a; i++) {
        for (int j = 0; j < col_a; j++) {
            float sum = 0; 
            for (int k = 0; k < col_b; k++) {
                sum += a[i*col_a+k] * b[k*col_b+j];
            }
            result[i*col_b+j] = sum;
        }
    }
}

void relu(float *result, float *a, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = (a[i] > 0) ? a[i] : 0;
    }
}

#endif