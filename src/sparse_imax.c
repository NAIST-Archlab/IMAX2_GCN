#include "../include/options.h"
#ifdef USE_IMAX2
#include <stdio.h>
#include <stdlib.h>
#include "../include/sparse.h"
#include "../include/emax6.h"
#include "../include/emax6lib.c"

void merge(Uint *b_num, int *b_value, Uint *a_num, int *a_value, Ull left, Ull mid, Ull right) {
    int i = left;
    int j = mid;
    int k = 0;
    int l;

    while (i < mid && j < right) {
        if (a_value[i] >= a_value[j]) {
            b_value[k] = a_value[i];
            b_num[k++] = a_num[i++];
        } else {
            b_value[k] = a_value[j];
            b_num[k++] = a_num[j++];
        }
    }

    if (i == mid) {
        while (j < right) {
            b_value[k] = a_value[j];
            b_num[k++] = a_num[j++];
        }
    } else {
        while (i < mid) {
            b_value[k] = a_value[i];
            b_num[k++] = a_num[i++];
        }
    }

    for (l=0; l < k; l++) {
        a_value[left + l] = b_value[l];
        a_num[left + l] = b_num[l];
    }
}

void merge_sort(Uint *num_buffer, int *value_buffer, Uint *num, int *value, Ull left, Ull right) {
    if (left == right || left == right-1) return;
    Ull mid = (left + right) / 2;
    merge_sort(num_buffer, value_buffer, num, value, left, mid);
    merge_sort(num_buffer, value_buffer, num, value, mid, right);
    merge(num_buffer, value_buffer, num, value, left, mid, right);
}

void trans_imax_format(IMAXSparseMatrix *imax_sp, SparseMatrix *sp) {
    imax_sp->row_size = sp->row_size;
    imax_sp->col_size = sp->col_size;
    imax_sp->row_num = (Uint*) malloc(sizeof(Uint)*sp->row_size);
    imax_sp->row_nnz = (int*) malloc(sizeof(int)*sp->row_size);

    for (int i=0; i < imax_sp->row_size; i++) {
        imax_sp->row_num[i] = i;
        imax_sp->row_nnz[i] = sp->row_p[i+1] - sp->row_p[i];
    }

    Uint *num_buffer = (Uint*) malloc(sizeof(Uint)*sp->row_size);
    int *value_buffer = (int*) malloc(sizeof(int)*sp->row_size);

    merge_sort(num_buffer, value_buffer, imax_sp->row_num, imax_sp->row_nnz, 0, sp->row_size);

    free(num_buffer);
    free(value_buffer);

    int row_p_blk = 46;
    int col_blk = 46;
    int blk_num = sp->row_size / row_p_blk;
    Ull new_nnz = 0;
    num_buffer = (Uint*) malloc(sizeof(Uint)*(blk_num+1));
    imax_sp->row_blk_size = row_p_blk;
    imax_sp->col_blk_size = col_blk;

    for (int i=0; i < blk_num; i++) {
        Ull max_nnz = imax_sp->row_nnz[i*row_p_blk];
        if ((max_nnz < col_blk) && (max_nnz != 0)) max_nnz = col_blk;
        else if ((max_nnz % col_blk) != 0) max_nnz += col_blk - (max_nnz % col_blk);
        num_buffer[i] = max_nnz;
        new_nnz += max_nnz*row_p_blk;
    }

    if ((imax_sp->row_size % row_p_blk) != 0) {
        Ull max_nnz = imax_sp->row_nnz[blk_num*row_p_blk];
        if ((max_nnz < col_blk) && (max_nnz != 0)) max_nnz = col_blk;
        else if ((max_nnz % col_blk) != 0) max_nnz += col_blk - (max_nnz % col_blk);
        num_buffer[blk_num] = max_nnz;
        new_nnz += max_nnz*(sp->row_size % row_p_blk);
    }

    imax_sp->val = (Uint*) malloc(sizeof(Uint)*(Ull)new_nnz);
    imax_sp->col_num = (Uint*) malloc(sizeof(Uint)*(Ull)new_nnz);
    imax_sp->nnz = new_nnz;

    Ull end = 0;
    for (Uint i=0; i < blk_num; i++) {
        for (Uint j=0; j < row_p_blk; j++) {
            Uint real_nnz = imax_sp->row_nnz[i*row_p_blk+j];
            Uint real_col_s = sp->row_p[imax_sp->row_num[i*row_p_blk+j]];
            imax_sp->row_nnz[i*row_p_blk+j] = num_buffer[i];
            for (Uint k=0; k < num_buffer[i]; k+=2) {
                if (real_nnz <= k) {
                    imax_sp->val[end+(row_p_blk*k)+(j*2)] = 0;
                    imax_sp->col_num[end+(row_p_blk*k)+(j*2)] = 0;
                    imax_sp->val[end+(row_p_blk*k)+(j*2)+1] = 0;
                    imax_sp->col_num[end+(row_p_blk*k)+(j*2)+1] = 0;
                } else {
                    imax_sp->val[end+(row_p_blk*k)+(j*2)] = *(Uint*)&(sp->val[real_col_s+k]);
                    imax_sp->col_num[end+(row_p_blk*k)+(j*2)] = sp->col_p[real_col_s+k];
                    if (real_nnz <= k+1) {
                        imax_sp->val[end+(row_p_blk*k)+(j*2)+1] = 0;
                        imax_sp->col_num[end+(row_p_blk*k)+(j*2)+1] = 0;
                    }
                    else {
                        imax_sp->val[end+(row_p_blk*k)+(j*2)+1] = *(Uint*)&(sp->val[real_col_s+k+1]);
                        imax_sp->col_num[end+(row_p_blk*k)+(j*2)+1] = sp->col_p[real_col_s+k+1];
                    }
                }
            }
        }
        end += num_buffer[i]*row_p_blk;
    }

    int i_iter;
    for (int i = blk_num*row_p_blk,i_iter=0;i < sp->row_size;i++,i_iter++) {
        int real_nnz = imax_sp->row_nnz[i];
        Uint real_col_s = sp->row_p[imax_sp->row_num[i]];
        imax_sp->row_nnz[i] = num_buffer[blk_num];
        for (Uint k=0; k < num_buffer[blk_num]; k+=2) {
            if (real_nnz <= k) {
                imax_sp->val[end+(row_p_blk*k)+(i_iter*2)] = 0;
                imax_sp->col_num[end+(row_p_blk*k)+(i_iter*2)] = 0;
                imax_sp->val[end+(row_p_blk*k)+(i_iter*2)+1] = 0;
                imax_sp->col_num[end+(row_p_blk*k)+(i_iter*2)+1] = 0;
            } else {
                imax_sp->val[end+(row_p_blk*k)+(i_iter*2)] = *(Uint*)&(sp->val[real_col_s+k]);
                imax_sp->col_num[end+(row_p_blk*k)+(i_iter*2)] = sp->col_p[real_col_s+k];
                if (real_nnz <= k+1) {
                    imax_sp->val[end+(row_p_blk*k)+(i_iter*2)+1] = 0;
                    imax_sp->col_num[end+(row_p_blk*k)+(i_iter*2)+1] = 0;
                } else {
                    imax_sp->val[end+(row_p_blk*k)+(i_iter*2)+1] = *(Uint*)&(sp->val[real_col_s+k+1]);
                    imax_sp->col_num[end+(row_p_blk*k)+(i_iter*2)+1] = sp->col_p[real_col_s+k+1];
                }
            }
        }
    }

    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < imax_sp->row_size; i++) imax_sp->row_num[i] *= 8;
    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < imax_sp->nnz; i++) imax_sp->col_num[i] *= 8;

    free(num_buffer);
}

void spmm(float* result, IMAXSparseMatrix *imax_sp_matrix, float* matrix, int mm_col) {
    Ull CHIP; Ull LOOP1, LOOP0; Ull INIT1, INIT0;
    Ull top, blk, blk_iter, end_sum, col_blk, row_blk;
    Ull AR[64][4]; /* output of EX in each unit */
    Ull BR[64][4][4]; /* output registers in each unit */
    Ull r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
    Ull r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
    Ull x0, x1, x2, x3;
    Ull cc0, cc1, cc2, cc3, ex0, ex1; Ull cofs, rofs, oofs, k, row_p, cofs1;

    #define NCHIP 4
    #define byte_size(type, num) ((Ull)sizeof(type)*(Ull)num)
    Ull *A_margin;
    Ull A_margin_tmp;

    Uint W = 4;
    size_t A_col_blk_size = imax_sp_matrix->col_blk_size;
    size_t A_row_blk_size = imax_sp_matrix->row_blk_size;
    size_t A_row_size = imax_sp_matrix->row_size;
    size_t A_col_size = imax_sp_matrix->col_size;
    size_t B_row_size = imax_sp_matrix->col_size;
    size_t B_col_size = mm_col;
    size_t A_nnz_size = 0;
    size_t B_col_blk = W*2*NCHIP;

    Ull cofs_init = (0-W*4*2*A_row_size)<<32|((0-W*4*2*B_row_size)&0xffffffff);
    Ull rofs_init = (0-1*8LL)<<32|((0-1*4LL)&0xffffffff);

    size_t B_col_blk_mul_B_row_size = B_col_blk*B_row_size;
    size_t A_row_size_mul_B_col_blk = A_row_size*B_col_blk;
    size_t A_row_size_mul_2_4_2 = A_row_size*2*4*2;
    size_t A_row_size_mul_8 = A_row_size*8;
    size_t A_row_size_mul_W_4_2 = (W*4*2*A_row_size)<<32|((W*4*2*B_row_size)&0xffffffff);
    size_t A_blk_size = imax_sp_matrix->col_blk_size * imax_sp_matrix->row_blk_size;
    size_t A_blk_size_mul_2 = A_blk_size * 2;
    size_t A_blk_size_mul_8 = A_blk_size * 8;

    typedef struct {Uint i[8]} Ui8;
    Uint *a_col_index[A_col_blk_size/2], *a[A_col_blk_size/2];
    Uint *b[NCHIP], *b0[NCHIP], *b1[NCHIP], *b2[NCHIP], *b3[NCHIP];
    Uint *c0[NCHIP], *c00[NCHIP], *c01[NCHIP], *c02[NCHIP], *c03[NCHIP];
    Uint *a_row_index;

    Uint tmp_cnt = 0;

    printf("<<<IMAX>>>\n");

    for (top=0; top < B_col_size/NCHIP; top+=B_col_blk) {
        for (row_blk=0,end_sum=0; row_blk < A_row_size; row_blk+=A_row_blk_size, end_sum+=A_nnz_size*A_row_blk_size) {
            if((A_nnz_size=imax_sp_matrix->row_nnz[row_blk])==0) break;
            a_row_index = (Uint*)imax_sp_matrix->row_num + row_blk;
            for (col_blk=0; col_blk < A_nnz_size; col_blk+=A_col_blk_size) {
                for (CHIP=0; CHIP<NCHIP; CHIP++) {
                    b[CHIP] = (Uint*)matrix + (CHIP*B_col_size/NCHIP + top)*B_row_size; 
                    b0[CHIP] = (Uint*)b[CHIP];
                    b1[CHIP] = (Uint*)b[CHIP] + B_row_size*2;
                    b2[CHIP] = (Uint*)b[CHIP] + B_row_size*4;
                    b3[CHIP] = (Uint*)b[CHIP] + B_row_size*6;
                }

                for (k=0; k < A_col_blk_size/2; k++) a[k] = (Uint*)imax_sp_matrix->val + end_sum + (col_blk*A_row_blk_size) + (2*k*A_row_blk_size);
                for (k=0; k < A_col_blk_size/2; k++) a_col_index[k] = (Uint*)imax_sp_matrix->col_num + end_sum + (col_blk*A_row_blk_size) + (2*k*A_row_blk_size);

                for (CHIP=0; CHIP<NCHIP; CHIP++) {
                    c0[CHIP] = (Uint*)result + (CHIP*B_col_size/NCHIP + top)*A_row_size;
                    c00[CHIP]= (Uint*)c0[CHIP];
                    c01[CHIP]= (Uint*)c0[CHIP] + A_row_size*2;
                    c02[CHIP]= (Uint*)c0[CHIP] + A_row_size*4;
                    c03[CHIP]= (Uint*)c0[CHIP] + A_row_size*6;
                }

                #define spmm_core1(r, rm1, offset) \
                            exe(OP_FMA, &AR[r][0], AR[rm1][0], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_FMA, &AR[r][1], AR[rm1][1], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_FMA, &AR[r][2], AR[rm1][2], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_FMA, &AR[r][3], AR[rm1][3], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            mop(OP_LDR,  3, &BR[r][0][1], (Ull)b0[CHIP], (Ull)offset,   MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                            mop(OP_LDR,  3, &BR[r][0][0], (Ull)b1[CHIP], (Ull)offset,   MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                            mop(OP_LDR,  3, &BR[r][1][1], (Ull)b2[CHIP], (Ull)offset,   MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                            mop(OP_LDR,  3, &BR[r][1][0], (Ull)b3[CHIP], (Ull)offset,   MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size)

                #define spmm_core1_load(r, rm1, rm2, a_col, offset) \
                            exe(OP_FMA, &AR[r][0], AR[rm1][0], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_FMA, &AR[r][1], AR[rm1][1], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_FMA, &AR[r][2], AR[rm1][2], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_FMA, &AR[r][3], AR[rm1][3], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            mop(OP_LDR,  3, &BR[r][0][1], (Ull)b0[CHIP], (Ull)offset,   MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                            mop(OP_LDR,  3, &BR[r][0][0], (Ull)b1[CHIP], (Ull)offset,   MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                            mop(OP_LDR,  3, &BR[r][1][1], (Ull)b2[CHIP], (Ull)offset,   MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                            mop(OP_LDR,  3, &BR[r][1][0], (Ull)b3[CHIP], (Ull)offset,   MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                            mop(OP_LDR,  3, &BR[r][2][1], (Ull)a[a_col],  (Ull)rofs, MSK_W1, (Ull)a[a_col], A_blk_size_mul_2, 0, 0, (Ull)NULL, A_blk_size_mul_2)

                #define spmm_core1_end(r, rm1, rm2, idx0, idx1, idx2, idx3, idx_base) \
                            exe(OP_FMA, &AR[r][0], AR[rm1][0], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_FMA, &AR[r][1], AR[rm1][1], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_FMA, &AR[r][2], AR[rm1][2], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_FMA, &AR[r][3], AR[rm1][3], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            mop(OP_LDR,  3, &BR[r][1][1], (Ull)a_col_index[idx0], (Ull)rofs,   MSK_W1, (Ull)a_col_index[idx_base], A_blk_size_mul_8,    0, 0, (Ull)NULL, A_blk_size_mul_8);\
                            mop(OP_LDR,  3, &BR[r][1][0], (Ull)a_col_index[idx1], (Ull)rofs,   MSK_W1, (Ull)a_col_index[idx_base], A_blk_size_mul_8,    0, 0, (Ull)NULL, A_blk_size_mul_8);\
                            mop(OP_LDR,  3, &BR[r][2][1], (Ull)a_col_index[idx2], (Ull)rofs,   MSK_W1, (Ull)a_col_index[idx_base], A_blk_size_mul_8,    0, 0, (Ull)NULL, A_blk_size_mul_8);\
                            mop(OP_LDR,  3, &BR[r][2][0], (Ull)a_col_index[idx3], (Ull)rofs,   MSK_W1, (Ull)a_col_index[idx_base], A_blk_size_mul_8,    0, 0, (Ull)NULL, A_blk_size_mul_8)

                #define spmm_core1_start(rp2, rp1, r, rm1, a_col) \
                            exe(OP_ADD, &r0, BR[rm1][1][1], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_ADD, &r1, BR[rm1][1][0], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_ADD, &r2, BR[rm1][2][1], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_ADD, &r3, BR[rm1][2][0], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            mop(OP_LDR,  3, &BR[rp1][0][1], (Ull)b0[CHIP], (Ull)r0,   MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                            mop(OP_LDR,  3, &BR[rp1][0][0], (Ull)b1[CHIP], (Ull)r0,   MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                            mop(OP_LDR,  3, &BR[rp1][1][1], (Ull)b2[CHIP], (Ull)r0,   MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                            mop(OP_LDR,  3, &BR[rp1][1][0], (Ull)b3[CHIP], (Ull)r0,   MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                            mop(OP_LDR,  3, &BR[rp1][2][1], (Ull)a[a_col],  (Ull)rofs, MSK_W1, (Ull)a[a_col], A_blk_size_mul_2, 0, 0, (Ull)NULL, A_blk_size_mul_2);\
                            exe(OP_FMA, &AR[rp2][0], AR[rm1][0], EXP_H3210, BR[rp1][2][1], EXP_H1010, BR[rp1][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_FMA, &AR[rp2][1], AR[rm1][1], EXP_H3210, BR[rp1][2][1], EXP_H1010, BR[rp1][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_FMA, &AR[rp2][2], AR[rm1][2], EXP_H3210, BR[rp1][2][1], EXP_H1010, BR[rp1][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_FMA, &AR[rp2][3], AR[rm1][3], EXP_H3210, BR[rp1][2][1], EXP_H1010, BR[rp1][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            mop(OP_LDR,  3, &BR[rp2][0][1], (Ull)b0[CHIP], (Ull)r0,   MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                            mop(OP_LDR,  3, &BR[rp2][0][0], (Ull)b1[CHIP], (Ull)r0,   MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                            mop(OP_LDR,  3, &BR[rp2][1][1], (Ull)b2[CHIP], (Ull)r0,   MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                            mop(OP_LDR,  3, &BR[rp2][1][0], (Ull)b3[CHIP], (Ull)r0,   MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size)

                #define spmm_core1_last_end(r, rm1, rm2, idx0, idx1, idx2, idx_base) \
                            exe(OP_FMA, &AR[r][0], AR[rm1][0], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_FMA, &AR[r][1], AR[rm1][1], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_FMA, &AR[r][2], AR[rm1][2], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_FMA, &AR[r][3], AR[rm1][3], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            mop(OP_LDR,  3, &BR[r][1][1], (Ull)a_col_index[idx0], (Ull)rofs,   MSK_W1, (Ull)a_col_index[idx_base], A_blk_size_mul_8,    0, 0, (Ull)NULL, A_blk_size_mul_8);\
                            mop(OP_LDR,  3, &BR[r][1][0], (Ull)a_col_index[idx1], (Ull)rofs,   MSK_W1, (Ull)a_col_index[idx_base], A_blk_size_mul_8,    0, 0, (Ull)NULL, A_blk_size_mul_8);\
                            mop(OP_LDR,  3, &BR[r][2][1], (Ull)a_col_index[idx2], (Ull)rofs,   MSK_W1, (Ull)a_col_index[idx_base], A_blk_size_mul_8,    0, 0, (Ull)NULL, A_blk_size_mul_8)

                #define spmm_core1_last_start(rp2, rp1, r, rm1, a_col) \
                            exe(OP_ADD, &r0, BR[rm1][1][1], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_ADD, &r1, BR[rm1][1][0], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_ADD, &r2, BR[rm1][2][1], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            mop(OP_LDR,  3, &BR[rp1][0][1], (Ull)b0[CHIP], (Ull)r0,   MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                            mop(OP_LDR,  3, &BR[rp1][0][0], (Ull)b1[CHIP], (Ull)r0,   MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                            mop(OP_LDR,  3, &BR[rp1][1][1], (Ull)b2[CHIP], (Ull)r0,   MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                            mop(OP_LDR,  3, &BR[rp1][1][0], (Ull)b3[CHIP], (Ull)r0,   MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                            mop(OP_LDR,  3, &BR[rp1][2][1], (Ull)a[a_col],  (Ull)rofs, MSK_W1, (Ull)a[a_col], A_blk_size_mul_2, 0, 0, (Ull)NULL, A_blk_size_mul_2);\
                            exe(OP_FMA, &AR[rp2][0], AR[rm1][0], EXP_H3210, BR[rp1][2][1], EXP_H1010, BR[rp1][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_FMA, &AR[rp2][1], AR[rm1][1], EXP_H3210, BR[rp1][2][1], EXP_H1010, BR[rp1][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_FMA, &AR[rp2][2], AR[rm1][2], EXP_H3210, BR[rp1][2][1], EXP_H1010, BR[rp1][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_FMA, &AR[rp2][3], AR[rm1][3], EXP_H3210, BR[rp1][2][1], EXP_H1010, BR[rp1][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            mop(OP_LDR,  3, &BR[rp2][0][1], (Ull)b0[CHIP], (Ull)r0,   MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                            mop(OP_LDR,  3, &BR[rp2][0][0], (Ull)b1[CHIP], (Ull)r0,   MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                            mop(OP_LDR,  3, &BR[rp2][1][1], (Ull)b2[CHIP], (Ull)r0,   MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);\
                            mop(OP_LDR,  3, &BR[rp2][1][0], (Ull)b3[CHIP], (Ull)r0,   MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size)
                            
                #define spmm_final(r, rm1) \
                            mop(OP_LDR, 3, &BR[r][0][1], (Ull)c00[CHIP], (Ull)r0, MSK_D0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 1, (Ull)NULL, A_row_size_mul_B_col_blk);\
                            mop(OP_LDR, 3, &BR[r][1][1], (Ull)c01[CHIP], (Ull)r0, MSK_D0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 1, (Ull)NULL, A_row_size_mul_B_col_blk);\
                            mop(OP_LDR, 3, &BR[r][2][1], (Ull)c02[CHIP], (Ull)r0, MSK_D0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 1, (Ull)NULL, A_row_size_mul_B_col_blk);\
                            mop(OP_LDR, 3, &BR[r][3][1], (Ull)c03[CHIP], (Ull)r0, MSK_D0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 1, (Ull)NULL, A_row_size_mul_B_col_blk);\
                            exe(OP_FAD, &AR[r][0], AR[rm1][0], EXP_H3210, BR[r][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_FAD, &AR[r][1], AR[rm1][1], EXP_H3210, BR[r][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_FAD, &AR[r][2], AR[rm1][2], EXP_H3210, BR[r][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            exe(OP_FAD, &AR[r][3], AR[rm1][3], EXP_H3210, BR[r][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);\
                            mop(OP_STR, 3, &AR[r][0], (Ull)r0, (Ull)c00[CHIP], MSK_D0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 0, (Ull)NULL, A_row_size_mul_B_col_blk);\
                            mop(OP_STR, 3, &AR[r][1], (Ull)r0, (Ull)c01[CHIP], MSK_D0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 0, (Ull)NULL, A_row_size_mul_B_col_blk);\
                            mop(OP_STR, 3, &AR[r][2], (Ull)r0, (Ull)c02[CHIP], MSK_D0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 0, (Ull)NULL, A_row_size_mul_B_col_blk);\
                            mop(OP_STR, 3, &AR[r][3], (Ull)r0, (Ull)c03[CHIP], MSK_D0, (Ull)c0[CHIP], A_row_size_mul_B_col_blk, 0, 0, (Ull)NULL, A_row_size_mul_B_col_blk)

            //EMAX5A begin spmm1 mapdist=0
                for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
                    for (INIT1=1,LOOP1=B_col_blk/(W*2),cofs=cofs_init; LOOP1--; INIT1=0) {
                        for (INIT0=1,LOOP0=A_row_blk_size,rofs=rofs_init; LOOP0--; INIT0=0) {
                            exe(OP_ADD, &cofs,  cofs,            EXP_H3210, INIT0?A_row_size_mul_W_4_2:0,     EXP_H3210, 0LL,    EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                            exe(OP_ADD, &rofs,  INIT0?rofs:rofs, EXP_H3210, (1*8LL)<<32|((1*4LL)&0xffffffff), EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL);
                            exe(OP_ADD, &cofs1, cofs,            EXP_H1010, 0,                                EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL);
                            exe(OP_ADD, &oofs,  cofs,            EXP_H3232, 0,                                EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,                  OP_NOP, 0LL);
                            mop(OP_LDR,  3, &BR[1][1][1], (Ull)a_col_index[0], (Ull)rofs, MSK_W1, (Ull)a_col_index[0], A_blk_size_mul_8, 0, 0, (Ull)NULL, A_blk_size_mul_8);
                            mop(OP_LDR,  3, &BR[1][1][0], (Ull)a_col_index[1], (Ull)rofs, MSK_W1, (Ull)a_col_index[0], A_blk_size_mul_8, 0, 0, (Ull)NULL, A_blk_size_mul_8);
                            mop(OP_LDR,  3, &BR[1][2][1], (Ull)a_col_index[2], (Ull)rofs, MSK_W1, (Ull)a_col_index[0], A_blk_size_mul_8, 0, 0, (Ull)NULL, A_blk_size_mul_8);
                            mop(OP_LDR,  3, &BR[1][2][0], (Ull)a_col_index[3], (Ull)rofs, MSK_W1, (Ull)a_col_index[0], A_blk_size_mul_8, 0, 0, (Ull)NULL, A_blk_size_mul_8);

                            exe(OP_ADD, &r0, BR[1][1][1], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                            exe(OP_ADD, &r1, BR[1][1][0], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                            exe(OP_ADD, &r2, BR[1][2][1], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                            exe(OP_ADD, &r3, BR[1][2][0], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                            mop(OP_LDR,  3, &BR[3][0][1], (Ull)b0[CHIP], (Ull)r0,   MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);
                            mop(OP_LDR,  3, &BR[3][0][0], (Ull)b1[CHIP], (Ull)r0,   MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);
                            mop(OP_LDR,  3, &BR[3][1][1], (Ull)b2[CHIP], (Ull)r0,   MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);
                            mop(OP_LDR,  3, &BR[3][1][0], (Ull)b3[CHIP], (Ull)r0,   MSK_W0, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);
                            mop(OP_LDR,  3, &BR[3][2][1], (Ull)a[0],  (Ull)rofs, MSK_W1, (Ull)a[0], A_blk_size_mul_2, 0, 0, (Ull)NULL, A_blk_size_mul_2);

                            exe(OP_FML, &AR[4][0], BR[3][2][1], EXP_H1010, BR[3][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                            exe(OP_FML, &AR[4][1], BR[3][2][1], EXP_H1010, BR[3][0][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                            exe(OP_FML, &AR[4][2], BR[3][2][1], EXP_H1010, BR[3][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                            exe(OP_FML, &AR[4][3], BR[3][2][1], EXP_H1010, BR[3][1][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                            mop(OP_LDR,  3, &BR[4][0][1], (Ull)b0[CHIP], (Ull)r0,   MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);
                            mop(OP_LDR,  3, &BR[4][0][0], (Ull)b1[CHIP], (Ull)r0,   MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);
                            mop(OP_LDR,  3, &BR[4][1][1], (Ull)b2[CHIP], (Ull)r0,   MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);
                            mop(OP_LDR,  3, &BR[4][1][0], (Ull)b3[CHIP], (Ull)r0,   MSK_W1, (Ull)b[CHIP], B_col_blk_mul_B_row_size,    0, 0, (Ull)NULL, B_col_blk_mul_B_row_size);

                            spmm_core1_load      (5,   4,  3,      1, r1);
                            spmm_core1           (6,   5,             r1);
                            spmm_core1_load      (7,   6,  5,      2, r2);
                            spmm_core1           (8,   7,             r2);
                            spmm_core1_load      (9,   8,  7,      3, r3);
                            spmm_core1           (10,  9,             r3);
                            spmm_core1_end       (11, 10,  9,      4,  5,  6,  7, 0);

                            spmm_core1_start     (14, 13, 12, 11,  4);
                            spmm_core1_load      (15, 14, 13,      5, r1);
                            spmm_core1           (16, 15,             r1);
                            spmm_core1_load      (17, 16, 15,      6, r2);
                            spmm_core1           (18, 17,             r2);
                            spmm_core1_load      (19, 18, 17,      7, r3);
                            spmm_core1           (20, 19,             r3);
                            spmm_core1_end       (21, 20, 19,      8,  9, 10, 11, 8);

                            spmm_core1_start     (24, 23, 22, 21,  8);
                            spmm_core1_load      (25, 24, 23,      9, r1);
                            spmm_core1           (26, 25,             r1);
                            spmm_core1_load      (27, 26, 25,     10, r2);
                            spmm_core1           (28, 27,             r2);
                            spmm_core1_load      (29, 28, 27,     11, r3);
                            spmm_core1           (30, 29,             r3);
                            spmm_core1_end       (31, 30, 29,     12, 13, 14, 15, 8);

                            spmm_core1_start     (34, 33, 32, 31, 12);
                            spmm_core1_load      (35, 34, 33,     13, r1);
                            spmm_core1           (36, 35,             r1);
                            spmm_core1_load      (37, 36, 35,     14, r2);
                            spmm_core1           (38, 37,             r2);
                            spmm_core1_load      (39, 38, 37,     15, r3);
                            spmm_core1           (40, 39,             r3);
                            spmm_core1_end       (41, 40, 39,     16, 17, 18, 19, 16);

                            spmm_core1_start     (44, 43, 42, 41, 16);
                            spmm_core1_load      (45, 44, 43,     17, r1);
                            spmm_core1           (46, 45,             r1);
                            spmm_core1_load      (47, 46, 45,     18, r2);
                            spmm_core1           (48, 47,             r2);
                            spmm_core1_load      (49, 48, 47,     19, r3);
                            spmm_core1           (50, 49,             r3);
                            spmm_core1_last_end  (51, 50, 49,     20, 21, 22, 16);

                            spmm_core1_last_start(54, 53, 52, 51, 20);
                            spmm_core1_load      (55, 54, 53,     21, r1);
                            spmm_core1           (56, 55,             r1);
                            spmm_core1_load      (57, 56, 55,     22, r2);
                            spmm_core1           (58, 57,             r2);

                            exe(OP_FMA, &AR[59][0], AR[58][0], EXP_H3210, BR[57][2][1], EXP_H3232, BR[58][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                            exe(OP_FMA, &AR[59][1], AR[58][1], EXP_H3210, BR[57][2][1], EXP_H3232, BR[58][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                            exe(OP_FMA, &AR[59][2], AR[58][2], EXP_H3210, BR[57][2][1], EXP_H3232, BR[58][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                            exe(OP_FMA, &AR[59][3], AR[58][3], EXP_H3210, BR[57][2][1], EXP_H3232, BR[58][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                            mop(OP_LDWR, 3, &BR[59][0][1], (Ull)a_row_index, (Ull)rofs, MSK_W0, (Ull)a_row_index, A_row_size, 0, 0, (Ull)NULL, A_row_size);

                            exe(OP_ADD, &r0, BR[59][0][1], EXP_H3210, oofs, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffLL, OP_NOP, 0LL);
                            spmm_final(62, 59);
                        }
                    }
                }
            //EMAX5A end
            }
            //EMAX5A drain_dirty_lmm
        }
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