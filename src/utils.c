#include "../include/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

double cal_time(struct timespec *end, struct timespec *start) {
    time_t result_sec;
    long result_nsec;

    result_sec = end->tv_sec - start->tv_sec;
    result_nsec = end->tv_nsec - start->tv_nsec;

    return result_sec + (double) (result_nsec / 1000000000.0);
}

int gcd(int a_, int b_) {
    int r, a  = a_, b  = b_;

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
    return a*b / gcd(a,b);
}

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

void imax_dense_format_init(IMAXDenseMatrix *imax_m, int row, int col, int row_padded, int col_padded, int row_blk, int col_blk) {
    imax_m->row_size = row;
    imax_m->col_size = col;
    imax_m->row_padded_size = row_padded;
    imax_m->col_padded_size = col_padded;
    imax_m->row_blk_size = row_blk;
    imax_m->col_blk_size = col_blk;
    imax_m->val = (Uint*) malloc(sizeof(Uint)*imax_m->row_padded_size*imax_m->col_padded_size);
    memset(imax_m->val, 0, sizeof(Uint)*imax_m->row_padded_size*imax_m->col_padded_size);
    printf("M Params: Orig(%d,%d) Padded(%d,%d) blk(%d,%d)\n", imax_m->row_size, imax_m->col_size, imax_m->row_padded_size, imax_m->col_padded_size, imax_m->row_blk_size, imax_m->col_blk_size);
}

void imax_dense_format_init_from_sparse(IMAXDenseMatrix *imax_m, IMAXSparseMatrix *imax_sp, int m_col, int m_col_blk_min) {
    imax_m->row_size = imax_sp->col_size;
    imax_m->col_size = m_col;
    
    imax_m->row_padded_size = imax_sp->col_padded_size;
    imax_m->row_blk_size = imax_sp->col_blk_size;
    imax_m->col_padded_size = imax_m->col_size + ((imax_m->col_size%m_col_blk_min) ? m_col_blk_min - imax_m->col_size%m_col_blk_min : 0);
    imax_m->col_blk_size = (imax_m->col_padded_size < ((int)sqrt(LMM_SIZE))) ? imax_m->col_padded_size : imax_m->col_padded_size/(imax_m->col_padded_size/(int)sqrt(LMM_SIZE));
    imax_m->val = (Uint*) malloc(sizeof(Uint)*imax_m->row_padded_size*imax_m->col_padded_size);
    memset(imax_m->val, 0, sizeof(Uint)*imax_m->row_padded_size*imax_m->col_padded_size);
    printf("M Params: Orig(%d,%d) Padded(%d,%d) blk(%d,%d)\n", imax_m->row_size, imax_m->col_size, imax_m->row_padded_size, imax_m->col_padded_size, imax_m->row_blk_size, imax_m->col_blk_size);
}

void convert_imax_dense_format(IMAXDenseMatrix *imax_m, Uint *m) {
    for (int b = 0; b < imax_m->row_padded_size / imax_m->row_blk_size; b++) {
        #ifdef USE_MP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < imax_m->row_blk_size; i++) {
            for (int j = 0; j < imax_m->col_size; j++) {
                if (imax_m->row_size > (b*imax_m->row_blk_size+i))
                    imax_m->val[(b*imax_m->col_padded_size*imax_m->row_blk_size)+((j/2)*(imax_m->row_blk_size*2)+(i*2)+(j%2))] = m[((b*imax_m->row_blk_size)+i)*imax_m->col_size+j];
            }
        }
    }
    printf("M -> IMAX_M: Orig(%d,%d) Padded(%d,%d) blk(%d,%d)\n", imax_m->row_size, imax_m->col_size, imax_m->row_padded_size, imax_m->col_padded_size, imax_m->row_blk_size, imax_m->col_blk_size);
}

void convert_dense_format(Uint *m, IMAXDenseMatrix *imax_m) {
    for (int b = 0; b < imax_m->row_padded_size / imax_m->row_blk_size; b++) {
        #ifdef USE_MP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < imax_m->row_blk_size; i++) {
            for (int j = 0; j < imax_m->col_size; j++) {
                if (imax_m->row_size > (b*imax_m->row_blk_size+i))
                    m[((b*imax_m->row_blk_size)+i)*imax_m->col_size+j] = imax_m->val[(b*imax_m->col_padded_size*imax_m->row_blk_size)+((j/2)*(imax_m->row_blk_size*2)+(i*2)+(j%2))];
            }
        }
    }
}

void imax_sparse_format_init(IMAXSparseMatrix *imax_sp, int row, int col, int sp_col_blk, int m_col_blk_min) {
    imax_sp->row_size = row;
    imax_sp->col_size = col;

    imax_sp->row_padded_size = row + ((row%m_col_blk_min) ? (m_col_blk_min - row%m_col_blk_min) : 0);
    imax_sp->row_blk_size = (imax_sp->row_padded_size < ((int)sqrt(LMM_SIZE))) ? imax_sp->row_padded_size : imax_sp->row_padded_size/(imax_sp->row_padded_size/((int)sqrt(LMM_SIZE)));
    imax_sp->col_padded_size = col + ((col%m_col_blk_min) ? (m_col_blk_min - col%m_col_blk_min) : 0);
    imax_sp->nnz_col_blk_size = sp_col_blk;
    imax_sp->col_blk_min = m_col_blk_min;
    imax_sp->col_blk_size = (imax_sp->col_padded_size < ((int)sqrt(LMM_SIZE))) ? imax_sp->col_padded_size : imax_sp->col_padded_size/(imax_sp->col_padded_size/((int)sqrt(LMM_SIZE)));
    imax_sp->sub = (IMAXSparseMatrixSub*) malloc(sizeof(IMAXSparseMatrixSub));
    imax_sp->sub->row_num = (Uint*) malloc(sizeof(Uint)*imax_sp->row_padded_size);
    imax_sp->sub->row_nnz = (int*) malloc(sizeof(int)*imax_sp->row_padded_size);
    imax_sp->sub->next = NULL;
    printf("SpM Parameters: Padded(%d,%d) blk(%d,%d) nnz_col_blk(%d)\n", imax_sp->row_padded_size, imax_sp->col_padded_size, imax_sp->row_blk_size, imax_sp->col_blk_size, imax_sp->nnz_col_blk_size);
}

void imax_sparse_format_next_init(IMAXSparseMatrixSub *imax_sp_sub, IMAXSparseMatrix *top_sp) {
    imax_sp_sub->next = (IMAXSparseMatrixSub*) malloc(sizeof(IMAXSparseMatrixSub));
    imax_sp_sub->next->row_num = (Uint*) malloc(sizeof(Uint)*top_sp->row_padded_size);
    imax_sp_sub->next->row_nnz = (int*) malloc(sizeof(int)*top_sp->row_padded_size);
    imax_sp_sub->next->next = NULL;
}

void convert_imax_sparse_format(IMAXSparseMatrix *imax_sp, SparseMatrix *sp) {
    IMAXSparseMatrixSub *p = imax_sp->sub;
    int row_p_blk = imax_sp->row_blk_size;
    int blk_num = imax_sp->row_padded_size / row_p_blk;
    int nnz_col_blk = imax_sp->nnz_col_blk_size;
    Uint *num_buffer = (Uint*) malloc(sizeof(Uint)*imax_sp->row_blk_size);
    int *value_buffer = (int*) malloc(sizeof(int)*imax_sp->row_blk_size);
    int *new_row_nnz = (int*) malloc(sizeof(int)*blk_num);

    for (int col_blk = 0; col_blk < (imax_sp->col_padded_size / imax_sp->col_blk_size); col_blk++) {
        int end = 0;
        int col_th_l = imax_sp->col_blk_size * col_blk;
        int col_th_h = imax_sp->col_blk_size * (col_blk+1);

        int l;
        for (l=0; l < imax_sp->row_size; l++) {
            p->row_num[l] = l;
            p->row_nnz[l] = 0;
            for (int j=sp->row_p[l]; j < sp->row_p[l+1]; j++) {
                if ((col_th_l <= sp->col_p[j]) && (col_th_h > sp->col_p[j])) p->row_nnz[l]++;
            }
        }

        for (; l < imax_sp->row_padded_size; l++) {
            p->row_num[l] = l;
            p->row_nnz[l] = 0;
        }

        for (int i=0; i < blk_num; i++) {
            int row_th_l = imax_sp->row_blk_size * i;
            merge_sort(num_buffer, value_buffer, p->row_num + row_th_l, p->row_nnz + row_th_l, 0, imax_sp->row_blk_size);
        }

        int new_nnz = 0;
        for (int i=0; i < blk_num; i++) {
            int max_nnz = p->row_nnz[i*row_p_blk];
            if (max_nnz) {
                if ((max_nnz < nnz_col_blk) && (max_nnz != 0)) max_nnz = nnz_col_blk;
                else if ((max_nnz % nnz_col_blk) != 0) max_nnz = max_nnz + nnz_col_blk - (max_nnz % nnz_col_blk);
            }
            new_row_nnz[i] = max_nnz;
            new_nnz += max_nnz*row_p_blk;
        }

        p->val = (Uint*) malloc(sizeof(Uint)*new_nnz);
        p->col_num = (Uint*) malloc(sizeof(Uint)*new_nnz);
        p->nnz = new_nnz;

        for (int i=0; i < blk_num; i++) {
            int row_th_l = imax_sp->row_blk_size*i;
            for (int j=0; j < imax_sp->row_blk_size; j++) {
                int real_nnz = p->row_nnz[row_th_l+j];
                int real_row = p->row_num[row_th_l+j];
                int real_col_s = sp->row_p[real_row];
                p->row_nnz[row_th_l+j] = new_row_nnz[i];

                int nnz_cnt = 0;
                for (int k=real_col_s; k < sp->row_p[real_row+1]; k++) {
                    if ((sp->col_p[k] < col_th_h) && (sp->col_p[k] >= col_th_l)) {
                        p->val[end + (imax_sp->row_blk_size*2)*(nnz_cnt/2)+(j*2)+(nnz_cnt%2)] = *(Uint*)&(sp->val[k]);
                        p->col_num[end + (imax_sp->row_blk_size*2)*(nnz_cnt/2)+(j*2)+(nnz_cnt%2)] = sp->col_p[k] - col_th_l;
                        nnz_cnt++;
                    }
                }

                for (int k=nnz_cnt; k < new_row_nnz[i]; k++) {
                    p->val[end+ (imax_sp->row_blk_size*2)*(k/2)+(j*2)+(k%2)] = 0;
                    p->col_num[end + (imax_sp->row_blk_size*2)*(k/2)+(j*2)+(k%2)] = 0;
                }
                p->row_num[row_th_l+j] -= row_th_l;
            }
            end += new_row_nnz[i]*imax_sp->row_blk_size;
        }

        #ifdef USE_MP
        #pragma omp parallel for
        #endif
        for (int k = 0; k < imax_sp->row_padded_size; k++) p->row_num[k] *= 8;

        #ifdef USE_MP
        #pragma omp parallel for
        #endif
        for (int k = 0; k < p->nnz; k++) p->col_num[k] *= 8;

        if ((col_blk+1) < (imax_sp->col_padded_size / imax_sp->col_blk_size)) {
            p->next = (IMAXSparseMatrixSub*) malloc(sizeof(IMAXSparseMatrixSub));
            p->next->row_num = (Uint*) malloc(sizeof(Uint)*imax_sp->row_padded_size);
            p->next->row_nnz = (int*) malloc(sizeof(int)*imax_sp->row_padded_size);
            p->next->next = NULL;
            p = p->next;
        }
    }

    free(num_buffer);
    free(value_buffer);
    free(new_row_nnz);

    printf("SpM -> IMAX_SpM: Padded(%d,%d) blk(%d,%d) col_blk_num(%d)\n", imax_sp->row_padded_size, imax_sp->col_padded_size, imax_sp->row_blk_size, imax_sp->col_blk_size, imax_sp->col_padded_size / imax_sp->col_blk_size);
}
