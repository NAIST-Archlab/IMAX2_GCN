#include "../include/utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double cal_time(struct timespec *end, struct timespec *start) {
    time_t result_sec;
    long result_nsec;

    result_sec = end->tv_sec - start->tv_sec;
    result_nsec = end->tv_nsec - start->tv_nsec;

    return result_sec*1000000.0 + (double)(result_nsec / 1000.0);
}

#ifdef EMAX6
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
    imax_m->col_blk_size = (imax_m->row_blk_size < MAX_COL_SIZE) ? LMM_SIZE/imax_m->row_blk_size : m_col_blk_min;
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
    imax_sp->row_padded_size = (row%imax_sp->row_blk_size) ? row + (imax_sp->col_blk_size - (row%imax_sp->col_blk_size)): row;
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

void imax_sparse_allocation(Uchar **membase, IMAXSparseMatrix *imax_sp) {
    #if defined(ARMZYNQ) && defined(EMAX6)
    int col_blk_num = imax_sp->col_padded_size / imax_sp->col_blk_size;
    Uchar is_membase_null = (*membase == NULL);
    Uint sparse_size = ((imax_sp->nnz * 2) + (imax_sp->row_padded_size * col_blk_num) + (imax_sp->nnz/imax_sp->nnz_col_blk_size)) * sizeof(Uint);
    printf("Will Allocate Memory Size: %luKiB\n", sparse_size / 1024);

    if (is_membase_null) {
        sysinit(membase, sparse_size, 32);
    } else {
        imax_add_alloc(membase, sparse_size, 32);
    }

    Uint *sp_tmp = (Uint *)*membase;
    sp_tmp = (Uint*)(((Uchar*)sp_tmp)-sizeof(Dll)-sparse_size);
    printf("IMAX Allocated Memory Base: %08x_%08x\n", (Uint)((Ull)sp_tmp >> 32), (Uint)sp_tmp);

    for (int i = 0; i < col_blk_num; i++) {
        printf("Sparse Input col[%03d] row_num Head: %08x_%08x\n", i, (Uint)((Ull)sp_tmp >> 32), (Uint)sp_tmp);
        memcpy(sp_tmp, imax_sp->sub[i]->row_num, (imax_sp->sub[i]->nnz/imax_sp->nnz_col_blk_size) * sizeof(Uint)); free((imax_sp->sub[i]->row_num)); imax_sp->sub[i]->row_num = sp_tmp; sp_tmp += (imax_sp->sub[i]->nnz/imax_sp->nnz_col_blk_size);
        printf("Sparse Input col[%03d] row_nnz Head: %08x_%08x\n", i, (Uint)((Ull)sp_tmp >> 32), (Uint)sp_tmp);
        memcpy(sp_tmp, imax_sp->sub[i]->row_nnz, imax_sp->row_padded_size * sizeof(Uint)); free((imax_sp->sub[i]->row_nnz)); imax_sp->sub[i]->row_nnz = sp_tmp; sp_tmp += imax_sp->row_padded_size;
        printf("Sparse Input col[%03d] col_num Head: %08x_%08x\n", i, (Uint)((Ull)sp_tmp >> 32), (Uint)sp_tmp);
        memcpy(sp_tmp, imax_sp->sub[i]->col_num, imax_sp->sub[i]->nnz     * sizeof(Uint)); free((imax_sp->sub[i]->col_num)); imax_sp->sub[i]->col_num = sp_tmp; sp_tmp += imax_sp->sub[i]->nnz;
        printf("Sparse Input col[%03d]     val Head: %08x_%08x\n", i, (Uint)((Ull)sp_tmp >> 32), (Uint)sp_tmp);
        memcpy(sp_tmp, imax_sp->sub[i]->val,     imax_sp->sub[i]->nnz     * sizeof(Uint)); free((imax_sp->sub[i]->val)    ); imax_sp->sub[i]->val = sp_tmp;     sp_tmp += imax_sp->sub[i]->nnz;
    }
    #endif
}

void imax_dense_allocation(Uchar **membase, IMAXDenseMatrix *imax_m) {
    Uint dense_size = (imax_m->row_padded_size * imax_m->col_padded_size) * sizeof(Uint);
    printf("Will Allocate Memory Size: %luKiB\n", dense_size / 1024);
    #if defined(ARMZYNQ) && defined(EMAX6)
    Uchar is_membase_null = (*membase == NULL);
    if (is_membase_null) {
        sysinit(membase, dense_size, 32);
    } else {
        imax_add_alloc(membase, dense_size, 32);
    }

    Uint *sp_tmp = (Uint *)*membase;
    sp_tmp = (Uint*)(((Uchar*)sp_tmp)-sizeof(Dll)-dense_size);

    printf("Dense Input  Head: %08x_%08x\n", (Uint)((Ull)sp_tmp >> 32), (Uint)sp_tmp);
    imax_m->val = sp_tmp;
    #else
    imax_m->val = (Uint *)malloc(dense_size);
    memset(imax_m->val, 0, dense_size);
    printf("Dense Input  Head: %08x_%08x\n", (Uint)((Ull)imax_m->val >> 32), (Uint)imax_m->val);
    #endif
}

void imax_sparse_deallocation(Uchar **membase, IMAXSparseMatrix *imax_sp) {
    int col_blk_num = (imax_sp->col_padded_size / imax_sp->col_blk_size);
    #if defined(ARMZYNQ) && defined(EMAX6)
    mem_release(membase, (
            (imax_sp->nnz * 2) +                                  // SpM Col and Data
            (imax_sp->row_padded_size * col_blk_num) +            // SpM Row
            (imax_sp->nnz/imax_sp->nnz_col_blk_size)
        ) *
        sizeof(Uint));
    #else
    for (int i = 0; i < col_blk_num; i++) {
        free(imax_sp->sub[i]->row_nnz);
        free(imax_sp->sub[i]->row_num);
        free(imax_sp->sub[i]->col_num);
        free(imax_sp->sub[i]->val);
        free(imax_sp->sub[i]);
    }
    #endif
}

void imax_dense_deallocation(Uchar **membase, IMAXDenseMatrix *imax_m) {
    #if defined(ARMZYNQ) && defined(EMAX6)
    mem_release(membase, (imax_m->row_padded_size * imax_m->col_padded_size) * sizeof(Uint));
    #else
    free(imax_m->val);
    #endif
}

void convert_imax_sparse_format(IMAXSparseMatrix *imax_sp, SparseMatrix *sp) {
    int row_p_blk = imax_sp->row_blk_size;
    int row_blk_num = imax_sp->row_padded_size / row_p_blk;
    int col_blk_num = imax_sp->col_padded_size / imax_sp->col_blk_size;
    int nnz_col_blk = imax_sp->nnz_col_blk_size;
    //Uint *num_buffer = (Uint *)malloc(sizeof(Uint) * imax_sp->row_blk_size);
    //int *value_buffer = (int *)malloc(sizeof(int) * imax_sp->row_blk_size);
    //int *new_row_nnz = (int *)malloc(sizeof(int) * row_blk_num);

    /*
    for (int i = 0; i < imax_sp->row_size; i++)
        for (int j = 0; j < col_blk_num; j++)
            imax_sp->sub[j]->row_num[i] = i;
    */

    for (int i = 0; i < imax_sp->row_size; i++) {
        for (int j = sp->row_p[i]; j < sp->row_p[i+1]; j++) {
            int col_blk = sp->col_p[j] / imax_sp->col_blk_size;
            imax_sp->sub[col_blk]->row_nnz[i]++;
        }
    }

    for (int i = imax_sp->row_size; i < imax_sp->row_padded_size; i++) {
        for (int j = 0; j < col_blk_num; j++) {
            //imax_sp->sub[j]->row_num[i] = i;
            imax_sp->sub[j]->row_nnz[i] = 0;
        }
    }

    for (int i = 0; i < col_blk_num; i++) {
        /*
        for (int j = 0; j < row_blk_num; j++) {
            int row_th_l = imax_sp->row_blk_size * j;
            merge_sort(num_buffer, value_buffer, imax_sp->sub[i]->row_num + row_th_l, imax_sp->sub[i]->row_nnz + row_th_l, 0, imax_sp->row_blk_size);
        }
        */

        int new_nnz = 0;
        for (int j = 0; j < imax_sp->row_padded_size; j++) {
            /*
        for (int j = 0; j < row_p_blk; j++) {
            int max_nnz = imax_sp->sub[i]->row_nnz[j * row_p_blk];
            if (max_nnz) {
                if ((max_nnz < nnz_col_blk) && (max_nnz != 0))
                    max_nnz = nnz_col_blk;
                else if ((max_nnz % nnz_col_blk) != 0)
                    max_nnz = max_nnz + nnz_col_blk - (max_nnz % nnz_col_blk);
            }
            new_row_nnz[j] = max_nnz;
            new_nnz += max_nnz * row_p_blk;
            */
           imax_sp->sub[i]->row_nnz[j] += (imax_sp->sub[i]->row_nnz[j]%nnz_col_blk) ? nnz_col_blk - (imax_sp->sub[i]->row_nnz[j]%nnz_col_blk) : 0;
           //if (!imax_sp->sub[i]->row_nnz[j]) imax_sp->sub[i]->row_nnz[j] = nnz_col_blk;
           new_nnz += imax_sp->sub[i]->row_nnz[j];
        }

        new_nnz += ((new_nnz/nnz_col_blk)%imax_sp->row_blk_size) ? (new_nnz/nnz_col_blk) - ((new_nnz/nnz_col_blk)%imax_sp->row_blk_size) : 0;

        imax_sp->sub[i]->val = (Uint *)malloc(sizeof(Uint) * new_nnz);
        imax_sp->sub[i]->col_num = (Uint *)malloc(sizeof(Uint) * new_nnz);
        imax_sp->sub[i]->row_num = (Uint *)malloc(sizeof(Uint) * new_nnz/nnz_col_blk);
        imax_sp->sub[i]->nnz = new_nnz;
        imax_sp->nnz += new_nnz;
        int nnz_row_size = new_nnz/nnz_col_blk;

        int nnz_cnt = 0;
        int nnz_blk_cnt = 0;
        int col_th_l = imax_sp->col_blk_size * i;
        int col_th_h = imax_sp->col_blk_size * (i + 1);

        for (int j = 0; j < row_blk_num; j++) {
            for (int k = 0; k < imax_sp->row_blk_size; k++) {
                int row_idx = j*imax_sp->row_blk_size + k;
                if (imax_sp->sub[i]->row_nnz[row_idx] > 0) {
                    int real_col_s = sp->row_p[row_idx];
                    int h = 0;

                    for (int l = real_col_s; l < sp->row_p[row_idx+1]; l++) {
                        if ((sp->col_p[l] < col_th_h) && (sp->col_p[l] >= col_th_l)) {
                            //printf("%d, ", sp->col_p[l] - col_th_l);
                            imax_sp->sub[i]->val[((nnz_row_size*2)*(h/2)) + (k*2) + (h%2)] = *(Uint *)&(sp->val[l]);
                            imax_sp->sub[i]->col_num[((nnz_row_size*2)*(h/2)) + (k*2) + (h%2)] = sp->col_p[l] - col_th_l;
                            h++;
                        }
                    }

                    for (int l = h; l < imax_sp->sub[i]->row_nnz[row_idx]; l++) {
                        imax_sp->sub[i]->val[((nnz_row_size*2)*(l/2)) + (k*2) + (l%2)] = 0;
                        imax_sp->sub[i]->col_num[((nnz_row_size*2)*(l/2)) + (k*2) + (l%2)] = 0;
                    }

                    for (int l = 0; l < imax_sp->sub[i]->row_nnz[row_idx]/nnz_col_blk; l++) {
                        imax_sp->sub[i]->row_num[nnz_blk_cnt++] = (Uint)row_idx;
                    }
                }
            }
        }
        /*
        for (int j = 0; j < row_blk_num; j++) {
            int row_th_l = imax_sp->row_blk_size * j;
            for (int k = 0; k < imax_sp->row_blk_size; k++) {
                int real_nnz = imax_sp->sub[i]->row_nnz[row_th_l + k];
                int real_row = imax_sp->sub[i]->row_num[row_th_l + k];
                if (real_nnz) {
                    int real_col_s = sp->row_p[real_row];
                    imax_sp->sub[i]->row_nnz[row_th_l + k] = new_row_nnz[j];

                    int nnz_cnt = 0;
                    for (int l = real_col_s; l < sp->row_p[real_row + 1]; l++) {
                        if ((sp->col_p[l] < col_th_h) && (sp->col_p[l] >= col_th_l)) {
                            imax_sp->sub[i]->val[end + (imax_sp->row_blk_size * 2) * (nnz_cnt / 2) + (k * 2) + (nnz_cnt % 2)] = *(Uint *)&(sp->val[l]);
                            imax_sp->sub[i]->col_num[end + (imax_sp->row_blk_size * 2) * (nnz_cnt / 2) + (k * 2) + (nnz_cnt % 2)] = sp->col_p[l] - col_th_l;
                            nnz_cnt++;
                        }
                    }

                    for (int l = nnz_cnt; l < new_row_nnz[j]; l++) {
                        imax_sp->sub[i]->val[end + (imax_sp->row_blk_size * 2) * (l / 2) + (k * 2) + (l % 2)] = 0;
                        imax_sp->sub[i]->col_num[end + (imax_sp->row_blk_size * 2) * (l / 2) + (k * 2) + (l % 2)] = 0;
                    }
                }
                imax_sp->sub[i]->row_num[row_th_l + k] -= row_th_l;
            }
            end += new_row_nnz[j] * imax_sp->row_blk_size;
        }
        */

        #ifdef USE_MP
        #pragma omp parallel for
        #endif
        for (int k = 0; k < imax_sp->sub[i]->nnz/nnz_col_blk; k++)
            imax_sp->sub[i]->row_num[k] *= 8;

        //for (int k = 0; k < imax_sp->sub[i]->nnz/nnz_col_blk; k++)
        //    printf("%d, ", imax_sp->sub[i]->row_num[k]);
        //printf("\n");

        #ifdef USE_MP
        #pragma omp parallel for
        #endif
        for (int k = 0; k < imax_sp->sub[i]->nnz; k++)
            imax_sp->sub[i]->col_num[k] *= 8;

        //for (int k = 0; k < imax_sp->sub[i]->nnz/10; k++)
            //printf("%d, ", imax_sp->sub[i]->col_num[k]/8);
        //printf("\n");

        //for (int k = 0; k < 500; k++)
        //    printf("%f, ", *(float*)&imax_sp->sub[i]->val[k]);
        //printf("\n");
        int row_num = imax_sp->nnz/imax_sp->nnz_col_blk_size;
        int val = 2;
        int prevVal = 0;
        while ((val = prime_factor(row_num, val)) != prevVal) {
            if (row_num/val < (LMM_SIZE/imax_sp->nnz_col_blk_size)) {imax_sp->sub[i]->nnz_row_blk_size = row_num/val; break;}
            prevVal = val;
        }
        if (prevVal == val) {imax_sp->sub[i]->nnz_row_blk_size = row_num/val;}
    }

    //int max_row_blk = LMM_SIZE/imax_sp->nnz_col_blk_size;
    //imax_sp->nnz_row_blk_size = imax_sp->nnz_row_blk_size - (imax_sp->nnz_row_blk_size);

    //free(num_buffer);
    //free(value_buffer);
    //free(new_row_nnz);

    Uint sparse_size = ((imax_sp->nnz * 2) + (imax_sp->row_padded_size * col_blk_num) + (imax_sp->nnz/imax_sp->nnz_col_blk_size)) * sizeof(Uint);
    printf("Allocated Memory Size: %uKiB\n", sparse_size / 1024);
    printf("SpM -> IMAX_SpM: Padded(%d,%d) blk(%d,%d) nnz_col_blk(%d)\n", imax_sp->row_padded_size, imax_sp->col_padded_size, imax_sp->row_blk_size, imax_sp->col_blk_size, imax_sp->nnz_col_blk_size);
    //exit(1);
}

#endif