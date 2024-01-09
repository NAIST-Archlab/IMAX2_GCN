// EMAX6/7 GCN Test Program            //
// utils.c                             //
//         Copyright (C) 2024 by NAIST //
//          Primary writer: Dohyun Kim //
//          kim.dohyun.kg7@is.naist.jp //
#include "../include/utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(EMAX7)
#include "../conv-c2d/emax7lib.c"
#elif defined(EMAX6)
#include "../conv-c2d/emax6lib.c"
#endif

char is_allocated = 0;

double cal_time(struct timespec *end, struct timespec *start) {
    time_t result_sec;
    long result_nsec;

    result_sec = end->tv_sec - start->tv_sec;
    result_nsec = end->tv_nsec - start->tv_nsec;

    return result_sec*1000000.0 + (double)(result_nsec / 1000.0);
}

#if defined(EMAX6) || defined(EMAX7)
Uchar *membase = NULL;
Uchar *prev = NULL;

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

Uchar* sysinit(Uint memsize, Uint alignment) {
#if defined(ARMZYNQ) && defined(EMAX6)
    if (emax6_open() == NULL) exit(1);
    membase = emax_info.ddr_mmap;
    {int i;for (i = 0; i < (memsize + sizeof(Dll) - 1) / sizeof(Dll); i++)*((Dll *)membase + i) = 0;}
    prev = (Uchar *)((Dll *)membase + (memsize + sizeof(Dll) - 1) / sizeof(Dll));
#elif defined(ARMZYNQ) && defined(EMAX7)
    if (emax7_open(1) == NULL) exit(1);
    membase = emax_info[0].ddr_mmap;
    {int i;for (i = 0; i < (memsize + sizeof(Dll) - 1) / sizeof(Dll); i++)*((Dll *)membase + i) = 0;}
    prev = (Uchar *)((Dll *)membase + (memsize + sizeof(Dll) - 1) / sizeof(Dll));
#elif __linux__ == 1
    posix_memalign(&membase, alignment, memsize);
#else
    membase = (void *)malloc(memsize + alignment);
    prev = membase;
    if ((Ull)membase & (Ull)(alignment - 1)) {membase = (void *)(((Ull)membase & ~(Ull)(alignment - 1)) + alignment);}
#endif

#if !defined(ARMZYNQ) && defined(EMAX6)
    emax_info.dma_phys = DMA_BASE2_PHYS; /* defined in emax6lib.h */
    emax_info.dma_mmap = emax_info.dma_phys;
    emax_info.reg_phys = REG_BASE2_PHYS; /* defined in emax6lib.h */
    emax_info.reg_mmap = emax_info.reg_phys;
    emax_info.lmm_phys = LMM_BASE2_PHYS;
    emax_info.lmm_mmap = emax_info.lmm_phys;
    emax_info.ddr_phys = membase;
    emax_info.ddr_mmap = emax_info.ddr_phys;
#elif !defined(ARMZYNQ) && defined(EMAX7)
    emax_info[0].dma_phys = DMA_BASE2_PHYS; /* defined in emax7lib.h */
    emax_info[0].dma_mmap = emax_info[0].dma_phys;
    emax_info[0].reg_phys = REG_BASE2_PHYS; /* defined in emax7lib.h */
    emax_info[0].reg_mmap = emax_info[0].reg_phys;
    emax_info[0].lmm_phys = LMM_BASE2_PHYS;
    emax_info[0].lmm_mmap = emax_info[0].lmm_phys;
    emax_info[0].ddr_phys = membase;
    emax_info[0].ddr_mmap = emax_info[0].ddr_phys;
#endif
#if (defined(ARMSIML) || defined(ARMZYNQ)) && defined(EMAX6)
    emax6.dma_ctrl = emax_info.dma_mmap;
    emax6.reg_ctrl = emax_info.reg_mmap;
    ((struct reg_ctrl *)emax6.reg_ctrl)->i[0].cmd = CMD_RESET;
#if defined(ARMZYNQ)
    usleep(1);
#endif
    switch (((struct reg_ctrl *)emax6.reg_ctrl)->i[0].stat >> 8 & 0xf) {
    case 3:
        EMAX_DEPTH = 64;
        break;
    case 2:
        EMAX_DEPTH = 32;
        break;
    case 1:
        EMAX_DEPTH = 16;
        break;
    default:
        EMAX_DEPTH = 8;
        break;
    }
    ((struct reg_ctrl *)emax6.reg_ctrl)->i[0].adtr = emax_info.ddr_mmap - emax_info.lmm_phys;
    ((struct reg_ctrl *)emax6.reg_ctrl)->i[0].dmrp = 0LL;
#endif
#if (defined(ARMSIML) || defined(ARMZYNQ)) && defined(EMAX7)
    emax7[0].dma_ctrl = emax_info[0].dma_mmap;
    emax7[0].reg_ctrl = emax_info[0].reg_mmap;
    ((struct reg_ctrl *)emax7[0].reg_ctrl)->i[0].cmd = CMD_RESET;
#if defined(ARMZYNQ)
    usleep(1);
#endif
    switch (((struct reg_ctrl *)emax7[0].reg_ctrl)->i[0].stat >> 8 & 0xf) {
    case 3:
        EMAX_DEPTH = 64;
        break;
    case 2:
        EMAX_DEPTH = 32;
        break;
    case 1:
        EMAX_DEPTH = 16;
        break;
    default:
        EMAX_DEPTH = 8;
        break;
    }
    ((struct reg_ctrl *)emax7[0].reg_ctrl)->i[0].adtr = emax_info[0].ddr_mmap - emax_info[0].lmm_phys;
    ((struct reg_ctrl *)emax7[0].reg_ctrl)->i[0].dmrp = 0LL;
#endif
    return membase;
}

Uchar* imax_alloc(Uint memsize, Uint alignment) {
    if (membase == NULL) {return sysinit(memsize, alignment);}
    else {
#if defined(ARMZYNQ) && (defined(EMAX6) || defined(EMAX7))
        membase = prev;
        {int i; for (i=0; i<(memsize+sizeof(Dll)-1)/sizeof(Dll); i++) *((Dll*)prev+i)=0;}
        prev = (Dll*)prev + ((memsize+sizeof(Dll)-1)/sizeof(Dll));
#elif __linux__ == 1
        posix_memalign(&membase, alignment, memsize);
#else
        membase = (void*)malloc(memsize+alignment);
        prev = membase;
        if ((Ull)membase & (Ull)(alignment-1)) {membase = (void*)(((Ull)membase & ~(Ull)(alignment-1))+alignment);}
#endif
        return membase;
    }
}

void imax_dealloc(Uint memsize, Uint alignment) {
    if (membase != NULL) {
#if defined(ARMZYNQ) && (defined(EMAX6) || defined(EMAX7))
        prev = (Dll*)prev - ((memsize+sizeof(Dll)-1)/sizeof(Dll));
#endif
    }
}

void imemcpy(Uint *dst, Uint *src, int words) {
    union {
        Uint i[4];
        Ull l[2];
        Dll d;
    } buf;

    Uint loop, i;
    if (words >= 1 && ((Ull)dst & sizeof(Uint))) { /* 4B-access odd */
        *dst++ = *src++;
        words--;
    }
    if (words >= 2 && ((Ull)dst & sizeof(Ull))) { /* 8B-access odd */
        if ((Ull)src & sizeof(Uint)) {
            buf.i[0] = *src++;
            buf.i[1] = *src++;
            *(Ull *)dst = buf.l[0];
        } else {
            *(Ull *)dst = *(Ull *)src;
            src += sizeof(Ull) / sizeof(Uint);
        }
        dst += sizeof(Ull) / sizeof(Uint);
        words -= 2;
    }

    if (loop = words / (sizeof(Dll) / sizeof(Uint))) {
        if ((Ull)src & sizeof(Uint)) {
            for (i = 0; i < loop; i++) {
                buf.i[0] = *src++;
                buf.i[1] = *src++;
                buf.i[2] = *src++;
                buf.i[3] = *src++;
                *(Dll *)dst = buf.d;
                dst += sizeof(Dll) / sizeof(Uint);
            }
        } else if ((Ull)src & sizeof(Ull)) {
            for (i = 0; i < loop; i++) {
                buf.l[0] = *(Ull *)src;
                src += sizeof(Ull) / sizeof(Uint);
                buf.l[1] = *(Ull *)src;
                src += sizeof(Ull) / sizeof(Uint);
                *(Dll *)dst = buf.d;
                dst += sizeof(Dll) / sizeof(Uint);
            }
        } else {
            for (i = 0; i < loop; i++) {
                *(Dll *)dst = *(Dll *)src;
                src += sizeof(Dll) / sizeof(Uint);
                dst += sizeof(Dll) / sizeof(Uint);
            }
        }
        words -= loop * (sizeof(Dll) / sizeof(Uint));
    }

    if (words >= 2) { /* 8B-access */
        if ((Ull)src & sizeof(Uint)) {
            buf.i[0] = *src++;
            buf.i[1] = *src++;
            *(Ull *)dst = buf.l[0];
        } else {
            *(Ull *)dst = *(Ull *)src;
            src += sizeof(Ull) / sizeof(Uint);
        }
        dst += sizeof(Ull) / sizeof(Uint);
        words -= 2;
    }
    if (words >= 1) { /* 4B-access */
        *dst++ = *src++;
        words--;
    }
}

void xmax_bzero(Uint *dst, int words) {
    Uint loop, i;
    if (words >= 1 && ((Ull)dst & sizeof(Uint))) { /* 4B-access odd */
        *dst++ = 0;
        words--;
    }
    if (words >= 2 && ((Ull)dst & sizeof(Ull))) { /* 8B-access odd */
        *(Ull *)dst = 0;
        dst += sizeof(Ull) / sizeof(Uint);
        words -= 2;
    }

    if (loop = words / (sizeof(Dll) / sizeof(Uint))) {
        for (i = 0; i < loop; i++) {
#if __AARCH64EL__ == 1
            *((Dll *)dst) = 0;
#else
            ((Dll *)dst)->u[0] = 0;
            ((Dll *)dst)->u[1] = 0;
#endif
            dst += sizeof(Dll) / sizeof(Uint);
        }
        words -= loop * (sizeof(Dll) / sizeof(Uint));
    }

    if (words >= 2) { /* 8B-access */
        *(Ull *)dst = 0;
        dst += sizeof(Ull) / sizeof(Uint);
        words -= 2;
    }
    if (words >= 1) { /* 4B-access */
        *dst++ = 0;
        words--;
    }
}


void imax_dense_format_init(IMAXDenseMatrix *imax_m, int row, int col, int row_padded, int col_padded, int row_blk, int col_blk) {
    imax_m->row_size = row;
    imax_m->col_size = col;
    imax_m->row_padded_size = row_padded;
    imax_m->col_padded_size = col_padded;
    imax_m->row_padded_size = (imax_m->row_padded_size < MM_H) ? MM_H: imax_m->row_padded_size;
    imax_m->col_padded_size = (imax_m->col_padded_size < MM_H) ? MM_H: imax_m->col_padded_size;
    imax_m->blk_row_size = (row_blk > row_padded) ? row_padded : row_blk;
    imax_m->blk_col_size = col_blk;
    printf("M Params: Orig(%d,%d) Padded(%d,%d) blk(%d,%d)\n", imax_m->row_size, imax_m->col_size, imax_m->row_padded_size, imax_m->col_padded_size, imax_m->blk_row_size, imax_m->blk_col_size);
}

void imax_dense_format_init_from_sparse(IMAXDenseMatrix *imax_m, IMAXSparseMatrix *imax_sp, int m_col, int m_col_blk_min) {
    imax_m->row_size = imax_sp->col_size;
    imax_m->col_size = m_col;

    imax_m->blk_row_size = imax_sp->blk_col_size;
    imax_m->row_padded_size = imax_sp->col_padded_size;
    int lmm_size_div_row_blk = (LMM_SIZE/2)/imax_m->blk_row_size; // LMMのサイズの最大値にすると構造上問題が発生するため、1/2にしている
    imax_m->blk_col_size = (imax_m->blk_row_size < MAX_COL_SIZE) ? lmm_size_div_row_blk - (lmm_size_div_row_blk%m_col_blk_min) : m_col_blk_min;
    imax_m->col_padded_size = (imax_m->col_size%MM_H) ? imax_m->col_size+(MM_H-(imax_m->col_size%MM_H)): imax_m->col_size;
    imax_m->row_padded_size = (imax_m->row_padded_size < MM_H) ? MM_H: imax_m->row_padded_size;
    imax_m->col_padded_size = (imax_m->col_padded_size < MM_H) ? MM_H: imax_m->col_padded_size;
    printf("M Params: Orig(%d,%d) Padded(%d,%d) blk(%d,%d)\n", imax_m->row_size, imax_m->col_size, imax_m->row_padded_size, imax_m->col_padded_size, imax_m->blk_row_size, imax_m->blk_col_size);
}

void imax_matrix_init_spmm(IMAXDenseMatrix *c, IMAXSparseMatrix *a, IMAXDenseMatrix *b, int matrix_flag) {
    if (matrix_flag == FIT_TO_SPARSE) {
        b->blk_row_size = a->blk_col_size;
        b->row_padded_size = a->col_padded_size;
        int lmm_size_div_row_blk = (LMM_SIZE/2)/b->blk_row_size; // LMMのサイズの最大値にすると構造上問題が発生するため、1/2にしている
        b->blk_col_size = (b->blk_row_size < MAX_COL_SIZE) ? lmm_size_div_row_blk - (lmm_size_div_row_blk%MM_MIN) : MM_MIN;
        b->col_padded_size = (b->col_size%MM_H) ? b->col_size+(MM_H-(b->col_size%MM_H)): b->col_size;
        b->row_padded_size = (b->row_padded_size < MM_H) ? MM_H: b->row_padded_size;
        b->col_padded_size = (b->col_padded_size < MM_H) ? MM_H: b->col_padded_size;
    }

    c->row_size = a->row_size;
    c->col_size = b->col_size;
    c->row_padded_size = a->row_padded_size;
    c->col_padded_size = b->col_padded_size;
    c->blk_row_size = a->blk_row_size;
    c->blk_col_size = b->blk_col_size;
    printf("Matrix B Params: Orig(%d,%d) Padded(%d,%d) blk(%d,%d)\n", b->row_size, b->col_size, b->row_padded_size, b->col_padded_size, b->blk_row_size, b->blk_col_size);
    printf("Matrix C Params: Orig(%d,%d) Padded(%d,%d) blk(%d,%d)\n", c->row_size, c->col_size, c->row_padded_size, c->col_padded_size, c->blk_row_size, c->blk_col_size);
}

void imax_matrix_init_mm(IMAXDenseMatrix *c, IMAXDenseMatrix *a, IMAXDenseMatrix *b, int matrix_flag) {
    if (matrix_flag == FIT_TO_DENSE) {
        int lmm_size_div_row_blk = (LMM_SIZE/2)/a->blk_row_size; // LMMのサイズの最大値にすると構造上問題が発生するため、1/2にしている
        a->blk_col_size = (a->blk_row_size < MAX_COL_SIZE) ? lmm_size_div_row_blk - (lmm_size_div_row_blk%MM_MIN) : MM_MIN;
        a->row_padded_size = (a->row_size%a->blk_row_size) ? a->row_size+(a->blk_row_size-(a->row_size%a->blk_row_size)): a->row_size;
        a->col_padded_size = (a->col_size%a->blk_col_size) ? a->col_size+(a->blk_col_size-(a->col_size%a->blk_col_size)): a->col_size;
        a->row_padded_size = (a->row_padded_size < MM_H) ? MM_H: a->row_padded_size;
        a->col_padded_size = (a->col_padded_size < MM_H) ? MM_H: a->col_padded_size;
    }
    b->blk_row_size = a->blk_col_size;
    b->blk_row_size = (b->blk_row_size < MM_H) ? MM_H: b->blk_row_size;
    b->blk_col_size = a->blk_col_size;
    b->row_padded_size = a->col_padded_size;
    b->col_padded_size = (b->col_size%b->blk_col_size) ? b->col_size+(b->blk_col_size-(b->col_size%b->blk_col_size)): b->col_size;
    b->row_padded_size = (b->row_padded_size < MM_H) ? MM_H: b->row_padded_size;
    b->col_padded_size = (b->col_padded_size < MM_H) ? MM_H: b->col_padded_size;

    c->row_size = a->row_size;
    c->col_size = b->col_size;
    c->row_padded_size = a->row_padded_size;
    c->col_padded_size = b->col_padded_size;
    c->blk_row_size = a->blk_row_size;
    c->blk_col_size = b->blk_col_size;
    printf("Matrix A Params: Orig(%d,%d) Padded(%d,%d) blk(%d,%d)\n", a->row_size, a->col_size, a->row_padded_size, a->col_padded_size, a->blk_row_size, a->blk_col_size);
    printf("Matrix B Params: Orig(%d,%d) Padded(%d,%d) blk(%d,%d)\n", b->row_size, b->col_size, b->row_padded_size, b->col_padded_size, b->blk_row_size, b->blk_col_size);
    printf("Matrix C Params: Orig(%d,%d) Padded(%d,%d) blk(%d,%d)\n", c->row_size, c->col_size, c->row_padded_size, c->col_padded_size, c->blk_row_size, c->blk_col_size);
}

void convert_imax_dense_format(IMAXDenseMatrix *imax_m, DenseMatrix *m) {
    for (int b = 0; b < (imax_m->row_padded_size/imax_m->blk_row_size); b++) {
        #ifdef USE_MP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < imax_m->blk_row_size; i++) {
            for (int j = 0; j < imax_m->col_size; j++) {
                if (imax_m->row_size > ((b*imax_m->blk_row_size)+i))
                    imax_m->val[(b*imax_m->col_padded_size*imax_m->blk_row_size) + (((j/2)*(imax_m->blk_row_size*2)) + (i*2) + (j%2))] = *(Uint*)&m->val[((b * imax_m->blk_row_size) + i) * imax_m->col_size + j];
            }
        }
    }
    printf("M -> IMAX_M: Orig(%d,%d) Padded(%d,%d) blk(%d,%d)\n", imax_m->row_size, imax_m->col_size, imax_m->row_padded_size, imax_m->col_padded_size, imax_m->blk_row_size, imax_m->blk_col_size);
}

void convert_dense_format(DenseMatrix *m, IMAXDenseMatrix *imax_m) {
    for (int b = 0; b < (imax_m->row_padded_size/imax_m->blk_row_size); b++) {
        #ifdef USE_MP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < imax_m->blk_row_size; i++) {
            for (int j = 0; j < imax_m->col_size; j++) {
                if (imax_m->row_size > ((b*imax_m->blk_row_size)+i))
                    m->val[((b*imax_m->blk_row_size)+i)*imax_m->col_size + j] = *(float*)&imax_m->val[(b*imax_m->col_padded_size*imax_m->blk_row_size) + (((j/2)*(imax_m->blk_row_size*2)) + (i*2) + (j%2))];
            }
        }
    }
    printf("IMAX_M -> M: Orig(%d,%d) Padded(%d,%d) blk(%d,%d)\n", imax_m->row_size, imax_m->col_size, imax_m->row_padded_size, imax_m->col_padded_size, imax_m->blk_row_size, imax_m->blk_col_size);
}

void imax_sparse_format_init(IMAXSparseMatrix *imax_sp, int row, int col, int sp_col_blk, int m_col_blk_min) {
    imax_sp->row_size = row;
    imax_sp->col_size = col;

    // Virtualized CHIP (64Units/2)
    #if !defined(HARD_UNIT32) && defined(UNIT32)
        #define ALL_CHIP 2*NCHIP
    #else 
        #define ALL_CHIP NCHIP
    #endif

    imax_sp->nnz_blk_col_size = sp_col_blk;
    imax_sp->blk_min_col = m_col_blk_min;
    imax_sp->nnz = 0;
    imax_sp->blk_row_size = (row < MAX_COL_SIZE) ? row + (row%2) : MAX_COL_SIZE;
    imax_sp->blk_row_size = (row/imax_sp->blk_row_size >= ALL_CHIP) ? imax_sp->blk_row_size : imax_sp->blk_row_size/ALL_CHIP;
    imax_sp->row_padded_size = (row%imax_sp->blk_row_size) ? row + (imax_sp->blk_row_size - (row%imax_sp->blk_row_size)): row;
    imax_sp->blk_col_size = (col < MAX_COL_SIZE) ? col + ((col%m_col_blk_min)?col:0) - (col%m_col_blk_min) : MAX_COL_SIZE;
    imax_sp->blk_col_size = (col/imax_sp->blk_col_size >= ALL_CHIP) ? imax_sp->blk_col_size : imax_sp->blk_col_size/ALL_CHIP;
    imax_sp->col_padded_size = (col%imax_sp->blk_col_size) ? col + (imax_sp->blk_col_size - (col%imax_sp->blk_col_size)): col;
    imax_sp->sub = (IMAXSparseMatrixSub *)malloc(sizeof(IMAXSparseMatrixSub) * (imax_sp->col_padded_size / imax_sp->blk_col_size));
    for (int i = 0; i < (imax_sp->col_padded_size / imax_sp->blk_col_size); i++) {
        imax_sp->sub[i].row_nnz = (Uint *)malloc(sizeof(Uint) * imax_sp->row_padded_size);
        memset(imax_sp->sub[i].row_nnz, 0, sizeof(Uint) * imax_sp->row_padded_size);
    }
    printf("SpM Parameters: Padded(%d,%d) blk(%d,%d) nnz_blk_col(%d)\n", imax_sp->row_padded_size, imax_sp->col_padded_size, imax_sp->blk_row_size, imax_sp->blk_col_size, imax_sp->nnz_blk_col_size);
}

#define ALIGN_SIZE sizeof(Dll)
#define PADDING_SIZE sizeof(Dll) * 10

void imax_sparse_allocation(IMAXSparseMatrix *imax_sp) {
    #if defined(ARMZYNQ) && (defined(EMAX6) || defined(EMAX7))
    Uint size = imax_sp->mem_size;
    Uint *sp_tmp = imax_alloc(size+PADDING_SIZE, 32);
    xmax_bzero(sp_tmp, (size+PADDING_SIZE)/sizeof(Uint));
    int blk_col_num = imax_sp->col_padded_size / imax_sp->blk_col_size;
    printf("IMAX Allocated Memory Base: %08x_%08x\n", (Uint)((Ull)sp_tmp >> 32), (Uint)sp_tmp);

    for (int i = 0; i < blk_col_num; i++) {
        printf("Sparse Input col[%03d] row_num Head: %08x_%08x\n", i, (Uint)((Ull)sp_tmp >> 32), (Uint)sp_tmp);
        imemcpy(sp_tmp, imax_sp->sub[i].row_num, imax_sp->sub[i].nnz/imax_sp->nnz_blk_col_size); free(imax_sp->sub[i].row_num); imax_sp->sub[i].row_num = sp_tmp; sp_tmp += (imax_sp->sub[i].nnz/imax_sp->nnz_blk_col_size);
        if ((Ull)sp_tmp%ALIGN_SIZE) sp_tmp += (ALIGN_SIZE - ((Ull)sp_tmp%ALIGN_SIZE))/sizeof(Uint);
        printf("Sparse Input col[%03d] row_nnz Head: %08x_%08x\n", i, (Uint)((Ull)sp_tmp >> 32), (Uint)sp_tmp);
        imemcpy(sp_tmp, imax_sp->sub[i].row_nnz,                      imax_sp->row_padded_size); free(imax_sp->sub[i].row_nnz); imax_sp->sub[i].row_nnz = sp_tmp; sp_tmp += imax_sp->row_padded_size;
        if ((Ull)sp_tmp%ALIGN_SIZE) sp_tmp += (ALIGN_SIZE - ((Ull)sp_tmp%ALIGN_SIZE))/sizeof(Uint);
        printf("Sparse Input col[%03d] col_num Head: %08x_%08x\n", i, (Uint)((Ull)sp_tmp >> 32), (Uint)sp_tmp);
        imemcpy(sp_tmp, imax_sp->sub[i].col_num,                           imax_sp->sub[i].nnz); free(imax_sp->sub[i].col_num); imax_sp->sub[i].col_num = sp_tmp; sp_tmp += imax_sp->sub[i].nnz;
        if ((Ull)sp_tmp%ALIGN_SIZE) sp_tmp += (ALIGN_SIZE - ((Ull)sp_tmp%ALIGN_SIZE))/sizeof(Uint);
        printf("Sparse Input col[%03d]     val Head: %08x_%08x\n", i, (Uint)((Ull)sp_tmp >> 32), (Uint)sp_tmp); 
        imemcpy(sp_tmp, imax_sp->sub[i].val,                               imax_sp->sub[i].nnz); free(    imax_sp->sub[i].val); imax_sp->sub[i].val     = sp_tmp; sp_tmp += imax_sp->sub[i].nnz;
        if ((Ull)sp_tmp%ALIGN_SIZE) sp_tmp += (ALIGN_SIZE - ((Ull)sp_tmp%ALIGN_SIZE))/sizeof(Uint);
    } 
    printf("The Sparse Matrix was allocated!\n");
    #endif
}

void imax_sparse_deallocation(IMAXSparseMatrix *imax_sp) {
    #if defined(ARMZYNQ) && (defined(EMAX6) || defined(EMAX7))
    imax_dealloc(imax_sp->mem_size+PADDING_SIZE, 32);
    #endif
}

void imax_dense_allocation(IMAXDenseMatrix *imax_m) {
    Uint size = imax_m->row_padded_size * imax_m->col_padded_size * sizeof(Uint);
    printf("Will Allocate Memory Size: %luKiB\n", size / 1024);
    #if defined(ARMZYNQ) && (defined(EMAX6) || defined(EMAX7))
    Uint *sp_tmp = imax_alloc(size, 32);
    xmax_bzero(sp_tmp, size/sizeof(Uint));
    imax_m->val = sp_tmp;
    printf("IMAX Allocated Memory Base: %08x_%08x\n", (Uint)((Ull)sp_tmp >> 32), (Uint)sp_tmp);
    #else
    imax_m->val = (Uint*) malloc(imax_m->row_padded_size * imax_m->col_padded_size * sizeof(Uint));
    memset(imax_m->val, 0, imax_m->row_padded_size * imax_m->col_padded_size * sizeof(Uint));
    #endif
    printf("Dense Input  Head: %08x_%08x\n", (Uint)((Ull)imax_m->val >> 32), (Uint)imax_m->val);
    printf("The Dense Matrix was allocated!\n");
}

void imax_dense_deallocation(IMAXDenseMatrix *imax_m) {
    #if defined(ARMZYNQ) && (defined(EMAX6) || defined(EMAX7))
    imax_dealloc(imax_m->row_padded_size * imax_m->col_padded_size * sizeof(Uint), 32);
    #else
    free(imax_m->val);
    #endif
}

void convert_imax_sparse_format(IMAXSparseMatrix *imax_sp, SparseMatrix *sp) {
    Uint sparse_size = 0;
    int blk_row_num = imax_sp->row_padded_size / imax_sp->blk_row_size;
    int blk_col_num = imax_sp->col_padded_size / imax_sp->blk_col_size;
    int nnz_blk_col_size = imax_sp->nnz_blk_col_size;

    for (int i = 0; i < imax_sp->row_size; i++) {
        for (int j = sp->row_p[i]; j < sp->row_p[i+1]; j++) {
            int col_blk = sp->col_p[j] / imax_sp->blk_col_size;
            imax_sp->sub[col_blk].row_nnz[i]++;
        }
    }

    for (int i = imax_sp->row_size; i < imax_sp->row_padded_size; i++) {
        for (int j = 0; j < blk_col_num; j++) {
            imax_sp->sub[j].row_nnz[i] = 0;
        }
    }

    for (int i = 0; i < blk_col_num; i++) {
        printf("col_blk_no: %d\n", i);
        int new_nnz = 0;
        int new_nnz_blk[blk_row_num];
        for (int j = 0; j < blk_row_num; j++) {new_nnz_blk[j] = 0;}
        for (int j = 0; j < imax_sp->row_padded_size; j++) {
           imax_sp->sub[i].row_nnz[j] += (imax_sp->sub[i].row_nnz[j]%nnz_blk_col_size) ? nnz_blk_col_size - (imax_sp->sub[i].row_nnz[j]%nnz_blk_col_size) : 0;
           new_nnz += imax_sp->sub[i].row_nnz[j];
           new_nnz_blk[j/imax_sp->blk_row_size] += imax_sp->sub[i].row_nnz[j];
        }

        int nnz_row_size = new_nnz/nnz_blk_col_size;
        int nnz_row_blk_size = (nnz_row_size < (MAX_COL_SIZE/4)) ? nnz_row_size : (MAX_COL_SIZE/4);
        imax_sp->sub[i].nnz_row_blk_size = nnz_row_blk_size;
        for (int j = 0; j < blk_row_num; j++) {
            int nnz_row_blk_row = new_nnz_blk[j]/nnz_blk_col_size;
            int nnz_row_blk_num = nnz_row_blk_row/(MAX_COL_SIZE/4);
            if (imax_sp->sub[i].nnz_row_blk_size*nnz_row_blk_num - nnz_row_blk_row < 0) nnz_row_blk_num++;
            int nnz_padded = (imax_sp->sub[i].nnz_row_blk_size*nnz_row_blk_num - nnz_row_blk_row)*nnz_blk_col_size;
            new_nnz += nnz_padded; imax_sp->sub[i].row_nnz[(imax_sp->blk_row_size*(j+1))-1] += nnz_padded;
            nnz_row_size += nnz_padded/nnz_blk_col_size;
        }

        imax_sp->nnz              += new_nnz;
        imax_sp->sub[i].nnz        = new_nnz;
        imax_sp->sub[i].val        = (Uint *)malloc(sizeof(Uint) * new_nnz);
        imax_sp->sub[i].col_num    = (Uint *)malloc(sizeof(Uint) * new_nnz);
        imax_sp->sub[i].row_num    = (Uint *)malloc(sizeof(Uint) * nnz_row_size);
        imax_sp->sub[i].row_blk    = (Uint *)malloc(sizeof(Uint) * (blk_row_num+1));
        imax_sp->sub[i].row_blk[0] = 0;
        sparse_size               += (new_nnz*2) + nnz_row_size + (blk_row_num+1) + imax_sp->row_padded_size;
        printf("nnz_size: %d\n", new_nnz);
        printf("row_num_size: %d\n", nnz_row_size);
        printf("row_blk: %d\n", blk_row_num);
        printf("nnz_row_blk_size: %d\n", nnz_row_blk_size);

        int nnz_blk_cnt = 0;
        int col_th_l = imax_sp->blk_col_size * i;
        int col_th_h = imax_sp->blk_col_size * (i + 1);
        float zero_f = 0;

        for (int j = 0; j < blk_row_num; j++) {
            for (int k = 0; k < imax_sp->blk_row_size; k++) {
                int row_idx = j*imax_sp->blk_row_size + k;
                if (imax_sp->sub[i].row_nnz[row_idx] > 0) {
                    int acc = 0;
                    int base = ((nnz_blk_cnt/nnz_row_blk_size)*nnz_row_blk_size*nnz_blk_col_size) + (nnz_blk_cnt%nnz_row_blk_size)*2;
                    if (row_idx < sp->row_size) {
                        for (int l = sp->row_p[row_idx]; l < sp->row_p[row_idx+1]; l++) {
                            if ((sp->col_p[l] < col_th_h) && (sp->col_p[l] >= col_th_l)) {
                                int nnz_blk_row_idx = acc/nnz_blk_col_size;
                                int nnz_blk_col_idx = acc%nnz_blk_col_size;
                                imax_sp->sub[i].val[base + ((nnz_row_blk_size*2)*(nnz_blk_col_idx/2)) + (nnz_blk_row_idx*2) + (nnz_blk_col_idx%2)] = *(Uint*)&(sp->val[l]);
                                imax_sp->sub[i].col_num[base + ((nnz_row_blk_size*2)*(nnz_blk_col_idx/2)) + (nnz_blk_row_idx*2) + (nnz_blk_col_idx%2)] = sp->col_p[l] - col_th_l;
                                acc++;
                            }
                        }
                    }

                    for (;acc < imax_sp->sub[i].row_nnz[row_idx]; acc++) {
                        int nnz_blk_row_idx = acc/nnz_blk_col_size;
                        int nnz_blk_col_idx = acc%nnz_blk_col_size;
                        imax_sp->sub[i].val[base + ((nnz_row_blk_size*2)*(nnz_blk_col_idx/2)) + (nnz_blk_row_idx*2) + (nnz_blk_col_idx%2)] = *(Uint*)&zero_f;
                        imax_sp->sub[i].col_num[base + ((nnz_row_blk_size*2)*(nnz_blk_col_idx/2)) + (nnz_blk_row_idx*2) + (nnz_blk_col_idx%2)] = 0;
                    }

                    for (int l = 0; l < imax_sp->sub[i].row_nnz[row_idx]/nnz_blk_col_size; l++) {
                        imax_sp->sub[i].row_num[nnz_blk_cnt++] = (Uint)k;
                    }
                }
            }
            imax_sp->sub[i].row_blk[j+1] = nnz_blk_cnt;
        }

        for (int k = 0; k < nnz_row_size; k++)
            imax_sp->sub[i].row_num[k] *= 8;

        for (int k = 0; k < imax_sp->sub[i].nnz; k++)
            imax_sp->sub[i].col_num[k] *= 8;
    }

    sparse_size *= sizeof(Uint);
    imax_sp->mem_size = sparse_size;
    printf("Allocated Memory Size: %uKiB\n", sparse_size / 1024);
    printf("SpM -> IMAX_SpM: Padded(%d,%d) blk(%d,%d) nnz_blk_col_size(%d)\n", imax_sp->row_padded_size, imax_sp->col_padded_size, imax_sp->blk_row_size, imax_sp->blk_col_size, imax_sp->nnz_blk_col_size);
}

#endif
