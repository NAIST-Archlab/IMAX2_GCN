#include "../include/utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char is_allocated = 0;

double cal_time(struct timespec *end, struct timespec *start) {
    time_t result_sec;
    long result_nsec;

    result_sec = end->tv_sec - start->tv_sec;
    result_nsec = end->tv_nsec - start->tv_nsec;

    return result_sec*1000000.0 + (double)(result_nsec / 1000.0);
}

#if defined(EMAX6) || defined(EMAX7)
int gcd(int a_, int b_) {
    int r, a = a_, b = b_;

    if (a < b) {
        r = a;
        a = b;
        b = r;
    }

    r = a % b;
    while (r != 0) {
        a = b;
        b = r;
        r = a % b;
    }

    return b;
}

int lcm(int a, int b) {
    return a * b / gcd(a, b);
}

int prime_factor(int val, int start) {
    for (int i = start+1; i*i <= val; i++) {
        if (val % i) continue;
        else return i;
    }
    return val; // val is a prime number
}

void merge(Uint *b_num, Uint *b_value, Uint *a_num, Uint *a_value, Ull left, Ull mid, Ull right) {
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

    for (l = 0; l < k; l++) {
        a_value[left + l] = b_value[l];
        a_num[left + l] = b_num[l];
    }
}

void merge_sort(Uint *num_buffer, int *value_buffer, Uint *num, Uint *value, Ull left, Ull right) {
    if (left == right || left == right - 1)
        return;
    Ull mid = (left + right) / 2;
    merge_sort(num_buffer, value_buffer, num, value, left, mid);
    merge_sort(num_buffer, value_buffer, num, value, mid, right);
    merge(num_buffer, value_buffer, num, value, left, mid, right);
}

void imax_dense_format_init(IMAXDenseMatrix *imax_m, int row, int col, int row_padded, int col_padded, int row_blk, int col_blk) {
    imax_m->row_size = row;
    imax_m->col_size = col;
    imax_m->row_padded_size = row_padded;
    imax_m->col_padded_size = col_padded;
    imax_m->row_padded_size = (imax_m->row_padded_size < MM_H) ? MM_H: imax_m->row_padded_size;
    imax_m->col_padded_size = (imax_m->col_padded_size < MM_H) ? MM_H: imax_m->col_padded_size;
    imax_m->row_blk_size = (row_blk > row_padded) ? row_padded : row_blk;
    imax_m->col_blk_size = col_blk;
    printf("M Params: Orig(%d,%d) Padded(%d,%d) blk(%d,%d)\n", imax_m->row_size, imax_m->col_size, imax_m->row_padded_size, imax_m->col_padded_size, imax_m->row_blk_size, imax_m->col_blk_size);
}

void imax_dense_format_init_from_sparse(IMAXDenseMatrix *imax_m, IMAXSparseMatrix *imax_sp, int m_col, int m_col_blk_min) {
    imax_m->row_size = imax_sp->col_size;
    imax_m->col_size = m_col;

    imax_m->row_blk_size = imax_sp->col_blk_size;
    imax_m->row_padded_size = imax_sp->col_padded_size;
    int lmm_size_div_row_blk = LMM_SIZE/imax_m->row_blk_size;
    imax_m->col_blk_size = (imax_m->row_blk_size < MAX_COL_SIZE) ? lmm_size_div_row_blk - (lmm_size_div_row_blk%m_col_blk_min) : m_col_blk_min;
    imax_m->col_padded_size = (imax_m->col_size%MM_H) ? imax_m->col_size+(MM_H-(imax_m->col_size%MM_H)): imax_m->col_size;
    imax_m->row_padded_size = (imax_m->row_padded_size < MM_H) ? MM_H: imax_m->row_padded_size;
    imax_m->col_padded_size = (imax_m->col_padded_size < MM_H) ? MM_H: imax_m->col_padded_size;
    printf("M Params: Orig(%d,%d) Padded(%d,%d) blk(%d,%d)\n", imax_m->row_size, imax_m->col_size, imax_m->row_padded_size, imax_m->col_padded_size, imax_m->row_blk_size, imax_m->col_blk_size);
}

void convert_imax_dense_format(IMAXDenseMatrix *imax_m, DenseMatrix *m) {
    for (int b = 0; b < (imax_m->row_padded_size/imax_m->row_blk_size); b++) {
        #ifdef USE_MP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < imax_m->row_blk_size; i++) {
            for (int j = 0; j < imax_m->col_size; j++) {
                if (imax_m->row_size > ((b*imax_m->row_blk_size)+i))
                    imax_m->val[(b*imax_m->col_padded_size*imax_m->row_blk_size) + (((j/2)*(imax_m->row_blk_size*2)) + (i*2) + (j%2))] = *(Uint*)&m->val[((b * imax_m->row_blk_size) + i) * imax_m->col_size + j];
            }
        }
    }
    printf("M -> IMAX_M: Orig(%d,%d) Padded(%d,%d) blk(%d,%d)\n", imax_m->row_size, imax_m->col_size, imax_m->row_padded_size, imax_m->col_padded_size, imax_m->row_blk_size, imax_m->col_blk_size);
}

void convert_dense_format(DenseMatrix *m, IMAXDenseMatrix *imax_m) {
    for (int b = 0; b < (imax_m->row_padded_size/imax_m->row_blk_size); b++) {
        #ifdef USE_MP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < imax_m->row_blk_size; i++) {
            for (int j = 0; j < imax_m->col_size; j++) {
                if (imax_m->row_size > ((b*imax_m->row_blk_size)+i))
                    m->val[((b*imax_m->row_blk_size)+i)*imax_m->col_size + j] = *(float*)&imax_m->val[(b*imax_m->col_padded_size*imax_m->row_blk_size) + (((j/2)*(imax_m->row_blk_size*2)) + (i*2) + (j%2))];
            }
        }
    }
    printf("IMAX_M -> M: Orig(%d,%d) Padded(%d,%d) blk(%d,%d)\n", imax_m->row_size, imax_m->col_size, imax_m->row_padded_size, imax_m->col_padded_size, imax_m->row_blk_size, imax_m->col_blk_size);
}

void imax_sparse_format_init(IMAXSparseMatrix *imax_sp, int row, int col, int sp_col_blk, int m_col_blk_min) {
    imax_sp->row_size = row;
    imax_sp->col_size = col;

    imax_sp->nnz_row_blk_size = LMM_SIZE/sp_col_blk;
    imax_sp->nnz_col_blk_size = sp_col_blk;
    imax_sp->col_blk_min = m_col_blk_min;
    imax_sp->nnz = 0;
    imax_sp->row_blk_size = (row < MAX_COL_SIZE) ? row + (row%2) : MAX_COL_SIZE;
    imax_sp->row_padded_size = (row%imax_sp->row_blk_size) ? row + (imax_sp->row_blk_size - (row%imax_sp->row_blk_size)): row;
    imax_sp->col_blk_size = (col < MAX_COL_SIZE) ? col + ((col%m_col_blk_min)?col:0) - (col%m_col_blk_min) : MAX_COL_SIZE;
    imax_sp->col_padded_size = (col%imax_sp->col_blk_size) ? col + (imax_sp->col_blk_size - (col%imax_sp->col_blk_size)): col;
    imax_sp->sub = (IMAXSparseMatrixSub **)malloc(sizeof(IMAXSparseMatrixSub *) * (imax_sp->col_padded_size / imax_sp->col_blk_size));
    for (int i = 0; i < (imax_sp->col_padded_size / imax_sp->col_blk_size); i++) {
        imax_sp->sub[i] = (IMAXSparseMatrixSub *)malloc(sizeof(IMAXSparseMatrixSub));
        imax_sp->sub[i]->row_nnz = (Uint *)malloc(sizeof(Uint) * imax_sp->row_padded_size);
        memset(imax_sp->sub[i]->row_nnz, 0, sizeof(Uint) * imax_sp->row_padded_size);
    }
    printf("SpM Parameters: Padded(%d,%d) blk(%d,%d) nnz_col_blk(%d)\n", imax_sp->row_padded_size, imax_sp->col_padded_size, imax_sp->row_blk_size, imax_sp->col_blk_size, imax_sp->nnz_col_blk_size);
}

void imax_gcn_allocation(IMAXSparseMatrix *imax_sp, IMAXDenseMatrix *imax_h, IMAXDenseMatrix *imax_spmm, IMAXDenseMatrix *imax_w, IMAXDenseMatrix *imax_mm) {
    Uint dense_size = (
        (imax_h->row_padded_size * imax_h->col_padded_size) +
        (imax_spmm->row_padded_size * imax_spmm->col_padded_size) +
        (imax_w->row_padded_size * imax_w->col_padded_size) +
        (imax_mm->row_padded_size * imax_mm->col_padded_size)
    ) * sizeof(Uint);
    #if defined(ARMZYNQ) && (defined(EMAX6) || defined(EMAX7))
    int col_blk_num = imax_sp->col_padded_size / imax_sp->col_blk_size;
    int row_blk_num = imax_sp->row_padded_size / imax_sp->row_blk_size;
    Uint sparse_size = ((imax_sp->nnz * 2) + (imax_sp->row_padded_size*col_blk_num) + (imax_sp->nnz/imax_sp->nnz_col_blk_size) + (row_blk_num+1)*col_blk_num) * sizeof(Uint);
    Uint all_size = dense_size + sparse_size;
    printf("Will Allocate Memory Size: %luKiB\n", all_size / 1024);

    Uint *sp_tmp;
    if (!is_allocated) {
        sp_tmp = sysinit(all_size, 32);
        printf("IMAX Allocated Memory Base: %08x_%08x\n", (Uint)((Ull)sp_tmp >> 32), (Uint)sp_tmp);

        for (int i = 0; i < col_blk_num; i++) {
            printf("Sparse Input col[%03d] row_num Head: %08x_%08x\n", i, (Uint)((Ull)sp_tmp >> 32), (Uint)sp_tmp);
            imemcpy(sp_tmp, imax_sp->sub[i]->row_num, (imax_sp->sub[i]->nnz/imax_sp->nnz_col_blk_size)); free((imax_sp->sub[i]->row_num)); imax_sp->sub[i]->row_num = sp_tmp; sp_tmp += (imax_sp->sub[i]->nnz/imax_sp->nnz_col_blk_size);
            printf("Sparse Input col[%03d] row_nnz Head: %08x_%08x\n", i, (Uint)((Ull)sp_tmp >> 32), (Uint)sp_tmp);
            imemcpy(sp_tmp, imax_sp->sub[i]->row_nnz, imax_sp->row_padded_size); free((imax_sp->sub[i]->row_nnz)); imax_sp->sub[i]->row_nnz = sp_tmp; sp_tmp += imax_sp->row_padded_size;
            printf("Sparse Input col[%03d] col_num Head: %08x_%08x\n", i, (Uint)((Ull)sp_tmp >> 32), (Uint)sp_tmp);
            imemcpy(sp_tmp, imax_sp->sub[i]->col_num, imax_sp->sub[i]->nnz); free((imax_sp->sub[i]->col_num)); imax_sp->sub[i]->col_num = sp_tmp; sp_tmp += imax_sp->sub[i]->nnz;
            printf("Sparse Input col[%03d]     val Head: %08x_%08x\n", i, (Uint)((Ull)sp_tmp >> 32), (Uint)sp_tmp);
            imemcpy(sp_tmp, imax_sp->sub[i]->val,     imax_sp->sub[i]->nnz); free((imax_sp->sub[i]->val)    ); imax_sp->sub[i]->val = sp_tmp;     sp_tmp += imax_sp->sub[i]->nnz;
            printf("Sparse Input col[%03d] row_blk Head: %08x_%08x\n", i, (Uint)((Ull)sp_tmp >> 32), (Uint)sp_tmp);
            imemcpy(sp_tmp, imax_sp->sub[i]->row_blk, row_blk_num+1); free((imax_sp->sub[i]->row_blk)); imax_sp->sub[i]->row_blk = sp_tmp; sp_tmp += (row_blk_num+1);
        }
        printf("The Sparse Matrix was allocated!\n");
        is_allocated = 1;
    } else {
        sp_tmp = imax_sp->sub[col_blk_num-1]->row_blk + row_blk_num + 1;
        xmax_bzero(sp_tmp, dense_size/sizeof(Uint));
    }
    imax_h->val = sp_tmp; sp_tmp += (imax_h->row_padded_size * imax_h->col_padded_size);
    imax_spmm->val = sp_tmp; sp_tmp += (imax_spmm->row_padded_size * imax_spmm->col_padded_size);
    imax_w->val = sp_tmp; sp_tmp += (imax_w->row_padded_size * imax_w->col_padded_size);
    imax_mm->val = sp_tmp; sp_tmp += (imax_mm->row_padded_size * imax_mm->col_padded_size);
    #else
    imax_h->val = (Uint*) malloc(imax_h->row_padded_size * imax_h->col_padded_size * sizeof(Uint));
    imax_spmm->val = (Uint*) malloc(imax_spmm->row_padded_size * imax_spmm->col_padded_size * sizeof(Uint));
    imax_w->val = (Uint*) malloc(imax_w->row_padded_size * imax_w->col_padded_size * sizeof(Uint));
    imax_mm->val = (Uint*) malloc(imax_mm->row_padded_size * imax_mm->col_padded_size * sizeof(Uint));
    memset(imax_h->val, 0, imax_h->row_padded_size * imax_h->col_padded_size * sizeof(Uint));
    memset(imax_spmm->val, 0, imax_spmm->row_padded_size * imax_spmm->col_padded_size * sizeof(Uint));
    memset(imax_w->val, 0, imax_w->row_padded_size * imax_w->col_padded_size * sizeof(Uint));
    memset(imax_mm->val, 0, imax_mm->row_padded_size * imax_mm->col_padded_size * sizeof(Uint));
    #endif
    printf("Dense Input  Head: %08x_%08x\n", (Uint)((Ull)imax_h->val >> 32), (Uint)imax_h->val);
    printf("Dense Input  Head: %08x_%08x\n", (Uint)((Ull)imax_spmm->val >> 32), (Uint)imax_spmm->val);
    printf("Dense Input  Head: %08x_%08x\n", (Uint)((Ull)imax_w->val >> 32), (Uint)imax_w->val);
    printf("Dense Input  Head: %08x_%08x\n", (Uint)((Ull)imax_mm->val >> 32), (Uint)imax_mm->val);
}

void convert_imax_sparse_format(IMAXSparseMatrix *imax_sp, SparseMatrix *sp) {
    int row_blk_num = imax_sp->row_padded_size / imax_sp->row_blk_size;
    int col_blk_num = imax_sp->col_padded_size / imax_sp->col_blk_size;
    int nnz_col_blk_size = imax_sp->nnz_col_blk_size;

    for (int i = 0; i < imax_sp->row_size; i++) {
        for (int j = sp->row_p[i]; j < sp->row_p[i+1]; j++) {
            int col_blk = sp->col_p[j] / imax_sp->col_blk_size;
            imax_sp->sub[col_blk]->row_nnz[i]++;
        }
    }

    for (int i = imax_sp->row_size; i < imax_sp->row_padded_size; i++) {
        for (int j = 0; j < col_blk_num; j++) {
            imax_sp->sub[j]->row_nnz[i] = 0;
        }
    }

    for (int i = 0; i < col_blk_num; i++) {
        int new_nnz = 0;
        for (int j = 0; j < imax_sp->row_padded_size; j++) {
           imax_sp->sub[i]->row_nnz[j] += (imax_sp->sub[i]->row_nnz[j]%nnz_col_blk_size) ? nnz_col_blk_size - (imax_sp->sub[i]->row_nnz[j]%nnz_col_blk_size) : 0;
           new_nnz += imax_sp->sub[i]->row_nnz[j];
        }

        int nnz_row_size = new_nnz/nnz_col_blk_size;
        int nnz_row_blk_size = (nnz_row_size < (MAX_COL_SIZE/4)) ? nnz_row_size : nnz_row_size/(nnz_row_size/(MAX_COL_SIZE/4));
        imax_sp->sub[i]->nnz_row_blk_size = nnz_row_blk_size;
        int nnz_row_blk_num = nnz_row_size/(MAX_COL_SIZE/4);
        new_nnz += (imax_sp->sub[i]->nnz_row_blk_size*nnz_row_blk_num - nnz_row_size)*nnz_col_blk_size;
        nnz_row_size = new_nnz/nnz_col_blk_size;
        imax_sp->sub[i]->row_nnz[imax_sp->row_padded_size-1] += (imax_sp->sub[i]->nnz_row_blk_size*nnz_row_blk_num - nnz_row_size)*nnz_col_blk_size;
        imax_sp->sub[i]->val = (Uint *)malloc(sizeof(Uint) * new_nnz);
        imax_sp->sub[i]->col_num = (Uint *)malloc(sizeof(Uint) * new_nnz);
        printf("nnz_size: %d\n", new_nnz);
        imax_sp->sub[i]->row_num = (Uint *)malloc(sizeof(Uint) * nnz_row_size);
        printf("row_num_size: %d\n", nnz_row_size);
        imax_sp->sub[i]->row_blk = (Uint *)malloc(sizeof(Uint) * (row_blk_num+1));
        printf("row_blk: %d\n", row_blk_num);
        printf("nnz_row_blk_size: %d\n", nnz_row_blk_size);
        imax_sp->sub[i]->row_blk[0] = 0;
        imax_sp->sub[i]->nnz = new_nnz;
        imax_sp->nnz += new_nnz;

        int nnz_blk_cnt = 0;
        int col_th_l = imax_sp->col_blk_size * i;
        int col_th_h = imax_sp->col_blk_size * (i + 1);

        for (int j = 0; j < row_blk_num; j++) {
            for (int k = 0; k < imax_sp->row_blk_size; k++) {
                int row_idx = j*imax_sp->row_blk_size + k;
                if (imax_sp->sub[i]->row_nnz[row_idx] > 0) {
                    int acc = 0;
                    int base = ((nnz_blk_cnt/nnz_row_blk_size)*nnz_row_blk_size*nnz_col_blk_size) + (nnz_blk_cnt%nnz_row_blk_size)*2;
                    for (int l = sp->row_p[row_idx]; l < sp->row_p[row_idx+1]; l++,acc++) {
                        if ((sp->col_p[l] < col_th_h) && (sp->col_p[l] >= col_th_l)) {
                            int nnz_blk_row_idx = acc/nnz_col_blk_size;
                            int nnz_blk_col_idx = acc%nnz_col_blk_size;
                            imax_sp->sub[i]->val[base + ((nnz_row_blk_size*2)*(nnz_blk_col_idx/2)) + (nnz_blk_row_idx*2) + (nnz_blk_col_idx%2)] = *(Uint*)&(sp->val[l]);
                            imax_sp->sub[i]->col_num[base + ((nnz_row_blk_size*2)*(nnz_blk_col_idx/2)) + (nnz_blk_row_idx*2) + (nnz_blk_col_idx%2)] = sp->col_p[l] - col_th_l;
                        }
                    }
                    for (;acc < imax_sp->sub[i]->row_nnz[row_idx]; acc++) {
                        int nnz_blk_row_idx = acc/nnz_col_blk_size;
                        int nnz_blk_col_idx = acc%nnz_col_blk_size;
                        imax_sp->sub[i]->val[base + ((nnz_row_blk_size*2)*(nnz_blk_col_idx/2)) + (nnz_blk_row_idx*2) + (nnz_blk_col_idx%2)] = (Uint)0;
                        imax_sp->sub[i]->col_num[base + ((nnz_row_blk_size*2)*(nnz_blk_col_idx/2)) + (nnz_blk_row_idx*2) + (nnz_blk_col_idx%2)] = (Uint)0;
                    }

                    for (int l = 0; l < imax_sp->sub[i]->row_nnz[row_idx]/nnz_col_blk_size; l++) {
                        imax_sp->sub[i]->row_num[nnz_blk_cnt++] = (Uint)k;
                    }
                }
            }
            imax_sp->sub[i]->row_blk[j+1] = nnz_blk_cnt;
        }

        for (int k = 0; k < nnz_row_size; k++)
            imax_sp->sub[i]->row_num[k] *= 8;

        for (int k = 0; k < imax_sp->sub[i]->nnz; k++)
            imax_sp->sub[i]->col_num[k] *= 8;

        //printf("%d\n", nnz_blk_cnt);

        //for (int k = 0; k < row_blk_num+1; k++)
            //printf("%d, ", imax_sp->sub[i]->row_blk[k]);
        //printf("\n");

        //for (int k = 0; k < imax_sp->sub[i]->nnz/nnz_col_blk_size; k++)
            //printf("%d, ", imax_sp->sub[i]->row_num[k]/8);
        //printf("\n");

        //for (int k = 0; k < imax_sp->sub[i]->nnz/10; k++)
            //printf("%d, ", imax_sp->sub[i]->col_num[k]/8);
        //printf("\n");

        //for (int k = 0; k < 500; k++)
        //    printf("%f, ", *(float*)&imax_sp->sub[i]->val[k]);
        //printf("\n");
        //int row_num = imax_sp->nnz/imax_sp->nnz_col_blk_size;
        //int val = 2;
        //int prevVal = 0;
        //while ((val = prime_factor(row_num, val)) != prevVal) {
            //if (row_num/val < (LMM_SIZE/imax_sp->nnz_col_blk_size)) {imax_sp->sub[i]->nnz_row_blk_size = row_num/val; break;}
            //prevVal = val;
        //}
        //if (prevVal == val) {imax_sp->sub[i]->nnz_row_blk_size = row_num/val;}
    }

    Uint sparse_size = ((imax_sp->nnz * 2) + (imax_sp->row_padded_size * col_blk_num) + (imax_sp->nnz/imax_sp->nnz_col_blk_size)) * sizeof(Uint);
    printf("Allocated Memory Size: %uKiB\n", sparse_size / 1024);
    printf("SpM -> IMAX_SpM: Padded(%d,%d) blk(%d,%d) nnz_col_blk_size(%d)\n", imax_sp->row_padded_size, imax_sp->col_padded_size, imax_sp->row_blk_size, imax_sp->col_blk_size, imax_sp->nnz_col_blk_size);
}

#endif