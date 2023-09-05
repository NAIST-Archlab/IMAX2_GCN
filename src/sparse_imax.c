#ifdef EMAX6
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/emax6.h"
#include "../include/emax6lib.c"
#include "../include/sparse.h"

void sysinit(Uchar **membase, Uint memsize, Uint alignment) {
#if defined(ARMZYNQ) && defined(EMAX6)
  if (emax6_open() == NULL)
    exit(1);
  *membase = emax_info.ddr_mmap;
  {int i; for (i=0; i<(memsize+sizeof(Dll)-1)/sizeof(Dll); i++) *((Dll*)*membase+i)=0;}
#elif __linux__ == 1
  posix_memalign(membase, alignment, memsize);
#else
  *membase = (void*)malloc(memsize+alignment);
  if ((Ull)*membase & (Ull)(alignment-1))
    *membase = (void*)(((Ull)*membase & ~(Ull)(alignment-1))+alignment);
#endif

#if !defined(ARMZYNQ) && defined(EMAX6)
  emax_info.dma_phys = DMA_BASE2_PHYS; /* defined in emax6lib.h */
  emax_info.dma_mmap = emax_info.dma_phys;
  emax_info.reg_phys = REG_BASE2_PHYS; /* defined in emax6lib.h */
  emax_info.reg_mmap = emax_info.reg_phys;
  emax_info.lmm_phys = LMM_BASE2_PHYS;
  emax_info.lmm_mmap = emax_info.lmm_phys;
  emax_info.ddr_phys = *membase;
  emax_info.ddr_mmap = emax_info.ddr_phys;
#endif
#if (defined(ARMSIML) || defined(ARMZYNQ)) && defined(EMAX6)
  emax6.dma_ctrl  = emax_info.dma_mmap;
  emax6.reg_ctrl  = emax_info.reg_mmap;
  ((struct reg_ctrl*)emax6.reg_ctrl)->i[0].cmd = CMD_RESET;
#if defined(ARMZYNQ)
  usleep(1);
#endif
  switch (((struct reg_ctrl*)emax6.reg_ctrl)->i[0].stat>>8 & 0xf) {
  case  3:EMAX_DEPTH = 64;break;
  case  2:EMAX_DEPTH = 32;break;
  case  1:EMAX_DEPTH = 16;break;
  default:EMAX_DEPTH =  8;break;
  }
  ((struct reg_ctrl*)emax6.reg_ctrl)->i[0].adtr = emax_info.ddr_mmap - emax_info.lmm_phys;
  ((struct reg_ctrl*)emax6.reg_ctrl)->i[0].dmrp = 0LL;
#endif
}

void imax_add_alloc(Uchar **membase, Uint memsize, Uint alignment) {
#if defined(ARMZYNQ) && defined(EMAX6)
  {int i; for (i=0; i<(memsize+sizeof(Dll)-1)/sizeof(Dll); i++) *((Dll*)*membase+i)=0;}
#elif __linux__ == 1
  posix_memalign(membase, alignment, memsize);
#else
  *membase = (void*)malloc(memsize+alignment);
  if ((Ull)*membase & (Ull)(alignment-1))
    *membase = (void*)(((Ull)*membase & ~(Ull)(alignment-1))+alignment);
#endif
}

void mem_release(Uchar **membase, Uint memsize) {
  #if defined(ARMZYNQ) && defined(EMAX6)
    {int i; for (i=0; i<(memsize+sizeof(Dll)-1)/sizeof(Dll); i++) *((Dll*)*membase+i)=0;}
  #else
  if(*membase != NULL){
    memset(*membase,0,memsize);
  }
  #endif
}

void spmm(IMAXDenseMatrix *result, IMAXSparseMatrix *imax_sp_matrix, IMAXDenseMatrix *matrix) {
    Ull CHIP;
    Ull LOOP1, LOOP0;
    Ull INIT1, INIT0;
    Ull blk, end_sum, nnz_col_blk, a_row_blk, b_col_blk;
    Ull AR[64][4];    /* output of EX in each unit */
    Ull BR[64][4][4]; /* output registers in each unit */
    Ull r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
    Ull r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
    Ull x0, x1, x2, x3;
    Ull cc0, cc1, cc2, cc3, ex0, ex1;
    Ull cofs, rofs, oofs, k;
    Ull cofs1;

    Uint W = 4;
    Ull A_row_blk_size = imax_sp_matrix->row_blk_size;
    Ull A_col_blk_size = imax_sp_matrix->col_blk_size;
    Ull A_nnz_col_blk_size = imax_sp_matrix->nnz_col_blk_size;
    Ull A_row_size = imax_sp_matrix->row_padded_size;
    Ull A_col_size = imax_sp_matrix->col_padded_size;
    Ull B_row_size = imax_sp_matrix->col_padded_size;
    Ull B_col_size = matrix->col_padded_size;
    Ull B_row_blk_size = matrix->row_blk_size;
    Ull B_col_blk_size = matrix->col_blk_size;

    Ull cofs_init = (0-W*4*2*A_row_blk_size)<<32|((0-W*4*2*B_row_blk_size)&0xffffffff);
    Ull rofs_init = (0-1*8LL)<<32|((0-1*4LL)&0xffffffff);

    Ull B_blk_size = B_row_blk_size * B_col_blk_size;
    Ull C_blk_size = A_row_blk_size * B_col_blk_size;
    Ull A_row_blk_size_mul_2 = A_row_blk_size*2;
    Ull A_row_blk_size_mul_8_2 = A_row_blk_size*8*2;
    Ull A_row_size_mul_W_4_2 = (W*4*2*A_row_blk_size)<<32|((W*4*2*B_row_blk_size)&0xffffffff);

    Uint *a_col_index[A_nnz_col_blk_size/2], *a[A_nnz_col_blk_size/2];
    Uint *a0, *a_col_index0;
    Uint *b[NCHIP], *b0[NCHIP], *b1[NCHIP], *b2[NCHIP], *b3[NCHIP];
    Uint *c0[NCHIP], *c00[NCHIP], *c01[NCHIP], *c02[NCHIP], *c03[NCHIP];
    Uint *a_row_index;
    Ull A_nnz_size = 0;
    Ull a_col_blk = 0;
    Ull a_col_blk_iter = 0;
    IMAXSparseMatrixSub **imax_sp_sub = imax_sp_matrix->sub;
    Uint *a_sub_col_head, *a_sub_head, *a_sub_nnz_head, *a_sub_row_head;
    Uint *b_head = (Uint*)matrix->val;
    Uint *c_head = (Uint*)result->val;

    printf("<<IMAX>>\n");

    // Select Column of A(=Row of B)
    for (a_col_blk = 0, a_col_blk_iter = 0; a_col_blk < A_col_size; a_col_blk += A_col_blk_size, a_col_blk_iter += 1) {
        a_sub_row_head = (Uint*)imax_sp_sub[a_col_blk_iter]->row_num;
        a_sub_nnz_head = (Uint*)imax_sp_sub[a_col_blk_iter]->row_nnz;
        a_sub_col_head = (Uint*)imax_sp_sub[a_col_blk_iter]->col_num;
        a_sub_head = (Uint*)imax_sp_sub[a_col_blk_iter]->val;

        // Select Row of A(=Row of C)
        for (a_row_blk = 0, end_sum = 0; a_row_blk < A_row_size; a_row_blk += A_row_blk_size, end_sum += A_nnz_size * A_row_blk_size) { // A_row_blk
            if ((A_nnz_size = a_sub_nnz_head[a_row_blk]) == 0) continue;
            a_row_index = (Uint*)a_sub_row_head + a_row_blk;

            // Select Column Block of of None-zero values of A
            for (nnz_col_blk = 0; nnz_col_blk < A_nnz_size; nnz_col_blk += A_nnz_col_blk_size) {
                for (k = 0; k < A_nnz_col_blk_size / 2; k++) a[k] = (Uint*)a_sub_head + end_sum + (nnz_col_blk * A_row_blk_size) + (2 * k * A_row_blk_size);
                for (k = 0; k < A_nnz_col_blk_size / 2; k++) a_col_index[k] = (Uint*)a_sub_col_head + end_sum + (nnz_col_blk * A_row_blk_size) + (2 * k * A_row_blk_size);

                // Select Column of B(= Column of C)
                for (b_col_blk = 0; b_col_blk < B_col_size / NCHIP; b_col_blk += B_col_blk_size) {
                    for (CHIP = 0; CHIP < NCHIP; CHIP++) {
                        b[CHIP] = (Uint*)b_head + a_col_blk * B_col_size + (CHIP * B_col_size / NCHIP + b_col_blk) * B_row_blk_size;
                        b0[CHIP] = (Uint*)b[CHIP];
                        b1[CHIP] = (Uint*)b[CHIP] + B_row_blk_size * 2;
                        b2[CHIP] = (Uint*)b[CHIP] + B_row_blk_size * 4;
                        b3[CHIP] = (Uint*)b[CHIP] + B_row_blk_size * 6;

                        c0[CHIP] = (Uint*)c_head + a_row_blk * B_col_size + (CHIP * B_col_size / NCHIP + b_col_blk) * A_row_blk_size;
                        c00[CHIP] = (Uint*)c0[CHIP];
                        c01[CHIP] = (Uint*)c0[CHIP] + A_row_blk_size * 2;
                        c02[CHIP] = (Uint*)c0[CHIP] + A_row_blk_size * 4;
                        c03[CHIP] = (Uint*)c0[CHIP] + A_row_blk_size * 6;
                    }

                    #define spmm_core1(r, rm1, offset) \
                                exe(OP_FMA, &AR[r][0], AR[rm1][0], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); \
                                exe(OP_FMA, &AR[r][1], AR[rm1][1], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); \
                                exe(OP_FMA, &AR[r][2], AR[rm1][2], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); \
                                exe(OP_FMA, &AR[r][3], AR[rm1][3], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); \
                                mop(OP_LDR, 3, &BR[r][0][1], (Ull)b0[CHIP], (Ull)offset, MSK_W1, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);     \
                                mop(OP_LDR, 3, &BR[r][0][0], (Ull)b1[CHIP], (Ull)offset, MSK_W1, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);     \
                                mop(OP_LDR, 3, &BR[r][1][1], (Ull)b2[CHIP], (Ull)offset, MSK_W1, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);     \
                                mop(OP_LDR, 3, &BR[r][1][0], (Ull)b3[CHIP], (Ull)offset, MSK_W1, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size)

                    #define spmm_core1_load(r, rm1, rm2, a_col, offset, a_idx_base) \
                                exe(OP_FMA, &AR[r][0], AR[rm1][0], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); \
                                exe(OP_FMA, &AR[r][1], AR[rm1][1], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); \
                                exe(OP_FMA, &AR[r][2], AR[rm1][2], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); \
                                exe(OP_FMA, &AR[r][3], AR[rm1][3], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); \
                                mop(OP_LDR, 3, &BR[r][0][1], (Ull)b0[CHIP], (Ull)offset, MSK_W0, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);     \
                                mop(OP_LDR, 3, &BR[r][0][0], (Ull)b1[CHIP], (Ull)offset, MSK_W0, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);     \
                                mop(OP_LDR, 3, &BR[r][1][1], (Ull)b2[CHIP], (Ull)offset, MSK_W0, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);     \
                                mop(OP_LDR, 3, &BR[r][1][0], (Ull)b3[CHIP], (Ull)offset, MSK_W0, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);     \
                                mop(OP_LDR, 3, &BR[r][2][1], (Ull)a[a_col], (Ull)rofs, MSK_W1, (Ull)a[a_idx_base], A_row_blk_size_mul_8_2, 0, 0, (Ull)NULL, A_row_blk_size_mul_8_2)

                    #define spmm_core1_end(r, rm1, rm2, idx0, idx1, idx2, idx3, idx_base) \
                                exe(OP_FMA, &AR[r][0], AR[rm1][0], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                          \
                                exe(OP_FMA, &AR[r][1], AR[rm1][1], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                          \
                                exe(OP_FMA, &AR[r][2], AR[rm1][2], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                          \
                                exe(OP_FMA, &AR[r][3], AR[rm1][3], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                          \
                                mop(OP_LDR, 3, &BR[r][1][1], (Ull)a_col_index[idx0], (Ull)rofs, MSK_W1, (Ull)a_col_index[idx_base], A_row_blk_size_mul_8_2, 0, 0, (Ull)NULL, A_row_blk_size_mul_8_2); \
                                mop(OP_LDR, 3, &BR[r][1][0], (Ull)a_col_index[idx1], (Ull)rofs, MSK_W1, (Ull)a_col_index[idx_base], A_row_blk_size_mul_8_2, 0, 0, (Ull)NULL, A_row_blk_size_mul_8_2); \
                                mop(OP_LDR, 3, &BR[r][2][1], (Ull)a_col_index[idx2], (Ull)rofs, MSK_W1, (Ull)a_col_index[idx_base], A_row_blk_size_mul_8_2, 0, 0, (Ull)NULL, A_row_blk_size_mul_8_2); \
                                mop(OP_LDR, 3, &BR[r][2][0], (Ull)a_col_index[idx3], (Ull)rofs, MSK_W1, (Ull)a_col_index[idx_base], A_row_blk_size_mul_8_2, 0, 0, (Ull)NULL, A_row_blk_size_mul_8_2)

                    #define spmm_core1_start(rp2, rp1, r, rm1, a_col, offset, a_idx_base) \
                                exe(OP_ADD, &r0, BR[rm1][1][1], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                                \
                                exe(OP_ADD, &r1, BR[rm1][1][0], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                                \
                                exe(OP_ADD, &r2, BR[rm1][2][1], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                                \
                                exe(OP_ADD, &r3, BR[rm1][2][0], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                                \
                                mop(OP_LDR, 3, &BR[rp1][0][1], (Ull)b0[CHIP], (Ull)offset, MSK_W0, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);                             \
                                mop(OP_LDR, 3, &BR[rp1][0][0], (Ull)b1[CHIP], (Ull)offset, MSK_W0, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);                             \
                                mop(OP_LDR, 3, &BR[rp1][1][1], (Ull)b2[CHIP], (Ull)offset, MSK_W0, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);                             \
                                mop(OP_LDR, 3, &BR[rp1][1][0], (Ull)b3[CHIP], (Ull)offset, MSK_W0, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);                             \
                                mop(OP_LDR, 3, &BR[rp1][2][1], (Ull)a[a_col], (Ull)rofs, MSK_W1, (Ull)a[a_idx_base], A_row_blk_size_mul_8_2, 0, 0, (Ull)NULL, A_row_blk_size_mul_8_2); \
                                exe(OP_FMA, &AR[rp2][0], AR[rm1][0], EXP_H3210, BR[rp1][2][1], EXP_H1010, BR[rp1][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                         \
                                exe(OP_FMA, &AR[rp2][1], AR[rm1][1], EXP_H3210, BR[rp1][2][1], EXP_H1010, BR[rp1][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                         \
                                exe(OP_FMA, &AR[rp2][2], AR[rm1][2], EXP_H3210, BR[rp1][2][1], EXP_H1010, BR[rp1][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                         \
                                exe(OP_FMA, &AR[rp2][3], AR[rm1][3], EXP_H3210, BR[rp1][2][1], EXP_H1010, BR[rp1][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                         \
                                mop(OP_LDR, 3, &BR[rp2][0][1], (Ull)b0[CHIP], (Ull)offset, MSK_W1, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);                             \
                                mop(OP_LDR, 3, &BR[rp2][0][0], (Ull)b1[CHIP], (Ull)offset, MSK_W1, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);                             \
                                mop(OP_LDR, 3, &BR[rp2][1][1], (Ull)b2[CHIP], (Ull)offset, MSK_W1, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);                             \
                                mop(OP_LDR, 3, &BR[rp2][1][0], (Ull)b3[CHIP], (Ull)offset, MSK_W1, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size)

                    #define spmm_core1_last_end(r, rm1, rm2, idx0, idx1, idx2, idx_base) \
                                exe(OP_FMA, &AR[r][0], AR[rm1][0], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                          \
                                exe(OP_FMA, &AR[r][1], AR[rm1][1], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                          \
                                exe(OP_FMA, &AR[r][2], AR[rm1][2], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                          \
                                exe(OP_FMA, &AR[r][3], AR[rm1][3], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                          \
                                mop(OP_LDR, 3, &BR[r][1][1], (Ull)a_col_index[idx0], (Ull)rofs, MSK_W1, (Ull)a_col_index[idx_base], A_row_blk_size_mul_8_2, 0, 0, (Ull)NULL, A_row_blk_size_mul_8_2); \
                                mop(OP_LDR, 3, &BR[r][1][0], (Ull)a_col_index[idx1], (Ull)rofs, MSK_W1, (Ull)a_col_index[idx_base], A_row_blk_size_mul_8_2, 0, 0, (Ull)NULL, A_row_blk_size_mul_8_2); \
                                mop(OP_LDR, 3, &BR[r][2][1], (Ull)a_col_index[idx2], (Ull)rofs, MSK_W1, (Ull)a_col_index[idx_base], A_row_blk_size_mul_8_2, 0, 0, (Ull)NULL, A_row_blk_size_mul_8_2)

                    #define spmm_core1_last_start(rp2, rp1, r, rm1, a_col, offset, a_idx_base) \
                                exe(OP_ADD, &r0, BR[rm1][1][1], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                                \
                                exe(OP_ADD, &r1, BR[rm1][1][0], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                                \
                                exe(OP_ADD, &r2, BR[rm1][2][1], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                                                \
                                mop(OP_LDR, 3, &BR[rp1][0][1], (Ull)b0[CHIP], (Ull)offset, MSK_W0, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);                             \
                                mop(OP_LDR, 3, &BR[rp1][0][0], (Ull)b1[CHIP], (Ull)offset, MSK_W0, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);                             \
                                mop(OP_LDR, 3, &BR[rp1][1][1], (Ull)b2[CHIP], (Ull)offset, MSK_W0, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);                             \
                                mop(OP_LDR, 3, &BR[rp1][1][0], (Ull)b3[CHIP], (Ull)offset, MSK_W0, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);                             \
                                mop(OP_LDR, 3, &BR[rp1][2][1], (Ull)a[a_col], (Ull)rofs, MSK_W1, (Ull)a[a_idx_base], A_row_blk_size_mul_8_2, 0, 0, (Ull)NULL, A_row_blk_size_mul_8_2); \
                                exe(OP_FMA, &AR[rp2][0], AR[rm1][0], EXP_H3210, BR[rp1][2][1], EXP_H1010, BR[rp1][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                         \
                                exe(OP_FMA, &AR[rp2][1], AR[rm1][1], EXP_H3210, BR[rp1][2][1], EXP_H1010, BR[rp1][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                         \
                                exe(OP_FMA, &AR[rp2][2], AR[rm1][2], EXP_H3210, BR[rp1][2][1], EXP_H1010, BR[rp1][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                         \
                                exe(OP_FMA, &AR[rp2][3], AR[rm1][3], EXP_H3210, BR[rp1][2][1], EXP_H1010, BR[rp1][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                         \
                                mop(OP_LDR, 3, &BR[rp2][0][1], (Ull)b0[CHIP], (Ull)offset, MSK_W1, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);                             \
                                mop(OP_LDR, 3, &BR[rp2][0][0], (Ull)b1[CHIP], (Ull)offset, MSK_W1, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);                             \
                                mop(OP_LDR, 3, &BR[rp2][1][1], (Ull)b2[CHIP], (Ull)offset, MSK_W1, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);                             \
                                mop(OP_LDR, 3, &BR[rp2][1][0], (Ull)b3[CHIP], (Ull)offset, MSK_W1, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size)

                    #define spmm_final(r, rm1, offset) \
                                mop(OP_LDR, 3, &BR[r][0][1], (Ull)c00[CHIP], (Ull)offset, MSK_W0, (Ull)c0[CHIP], C_blk_size, 0, 1, (Ull)NULL, C_blk_size); \
                                mop(OP_LDR, 3, &BR[r][1][1], (Ull)c01[CHIP], (Ull)offset, MSK_W0, (Ull)c0[CHIP], C_blk_size, 0, 1, (Ull)NULL, C_blk_size); \
                                mop(OP_LDR, 3, &BR[r][2][1], (Ull)c02[CHIP], (Ull)offset, MSK_W0, (Ull)c0[CHIP], C_blk_size, 0, 1, (Ull)NULL, C_blk_size); \
                                mop(OP_LDR, 3, &BR[r][3][1], (Ull)c03[CHIP], (Ull)offset, MSK_W0, (Ull)c0[CHIP], C_blk_size, 0, 1, (Ull)NULL, C_blk_size); \
                                exe(OP_FAD, &AR[r][0], AR[rm1][0], EXP_H3210, BR[r][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           \
                                exe(OP_FAD, &AR[r][1], AR[rm1][1], EXP_H3210, BR[r][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           \
                                exe(OP_FAD, &AR[r][2], AR[rm1][2], EXP_H3210, BR[r][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           \
                                exe(OP_FAD, &AR[r][3], AR[rm1][3], EXP_H3210, BR[r][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);           \
                                mop(OP_STR, 3, &AR[r][0], (Ull)offset, (Ull)c00[CHIP], MSK_D0, (Ull)c0[CHIP], C_blk_size, 0, 1, (Ull)NULL, C_blk_size);    \
                                mop(OP_STR, 3, &AR[r][1], (Ull)offset, (Ull)c01[CHIP], MSK_D0, (Ull)c0[CHIP], C_blk_size, 0, 1, (Ull)NULL, C_blk_size);    \
                                mop(OP_STR, 3, &AR[r][2], (Ull)offset, (Ull)c02[CHIP], MSK_D0, (Ull)c0[CHIP], C_blk_size, 0, 1, (Ull)NULL, C_blk_size);    \
                                mop(OP_STR, 3, &AR[r][3], (Ull)offset, (Ull)c03[CHIP], MSK_D0, (Ull)c0[CHIP], C_blk_size, 0, 1, (Ull)NULL, C_blk_size)

//EMAX5A begin spmm1 mapdist=0
                    for (CHIP=0;CHIP<NCHIP;CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
                        for (INIT1=1,LOOP1=B_col_blk_size/(W*2),cofs=cofs_init;LOOP1--;INIT1=0) {
                            for (INIT0=1,LOOP0=A_row_blk_size,rofs=rofs_init;LOOP0--;INIT0=0) {
                                exe(OP_ADD, &cofs,            cofs, EXP_H3210,     INIT0?A_row_size_mul_W_4_2:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP,                  0LL, OP_NOP, 0LL);
                                exe(OP_ADD, &rofs, INIT0?rofs:rofs, EXP_H3210, (1*8LL)<<32|((1*4LL)&0xffffffff), EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL);
                                exe(OP_ADD, &cofs1,           cofs, EXP_H1010, 0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                                exe(OP_ADD, &oofs,            cofs, EXP_H3232, 0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                                mop(OP_LDR, 3, &BR[1][1][1], (Ull)a_col_index[0], (Ull)rofs, MSK_W1, (Ull)a_col_index[0], A_row_blk_size_mul_8_2, 0, 0, (Ull)NULL, A_row_blk_size_mul_8_2);
                                mop(OP_LDR, 3, &BR[1][1][0], (Ull)a_col_index[1], (Ull)rofs, MSK_W1, (Ull)a_col_index[0], A_row_blk_size_mul_8_2, 0, 0, (Ull)NULL, A_row_blk_size_mul_8_2);
                                mop(OP_LDR, 3, &BR[1][2][1], (Ull)a_col_index[2], (Ull)rofs, MSK_W1, (Ull)a_col_index[0], A_row_blk_size_mul_8_2, 0, 0, (Ull)NULL, A_row_blk_size_mul_8_2);
                                mop(OP_LDR, 3, &BR[1][2][0], (Ull)a_col_index[3], (Ull)rofs, MSK_W1, (Ull)a_col_index[0], A_row_blk_size_mul_8_2, 0, 0, (Ull)NULL, A_row_blk_size_mul_8_2);

                                exe(OP_ADD, &r0, BR[1][1][1], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                                exe(OP_ADD, &r1, BR[1][1][0], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                                exe(OP_ADD, &r2, BR[1][2][1], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                                exe(OP_ADD, &r3, BR[1][2][0], EXP_H3210, cofs1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                                mop(OP_LDR, 3, &BR[3][0][1], (Ull)b0[CHIP], (Ull)r0, MSK_W0, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);
                                mop(OP_LDR, 3, &BR[3][0][0], (Ull)b1[CHIP], (Ull)r0, MSK_W0, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);
                                mop(OP_LDR, 3, &BR[3][1][1], (Ull)b2[CHIP], (Ull)r0, MSK_W0, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);
                                mop(OP_LDR, 3, &BR[3][1][0], (Ull)b3[CHIP], (Ull)r0, MSK_W0, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);
                                mop(OP_LDR, 3, &BR[3][2][1], (Ull)a[0], (Ull)rofs, MSK_W1, (Ull)a[0], A_row_blk_size_mul_8_2, 0, 0, (Ull)NULL, A_row_blk_size_mul_2);

                                exe(OP_FML, &AR[4][0], BR[3][2][1], EXP_H1010, BR[3][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                                exe(OP_FML, &AR[4][1], BR[3][2][1], EXP_H1010, BR[3][0][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                                exe(OP_FML, &AR[4][2], BR[3][2][1], EXP_H1010, BR[3][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                                exe(OP_FML, &AR[4][3], BR[3][2][1], EXP_H1010, BR[3][1][0], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                                mop(OP_LDR, 3, &BR[4][0][1], (Ull)b0[CHIP], (Ull)r0, MSK_W1, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);
                                mop(OP_LDR, 3, &BR[4][0][0], (Ull)b1[CHIP], (Ull)r0, MSK_W1, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);
                                mop(OP_LDR, 3, &BR[4][1][1], (Ull)b2[CHIP], (Ull)r0, MSK_W1, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);
                                mop(OP_LDR, 3, &BR[4][1][0], (Ull)b3[CHIP], (Ull)r0, MSK_W1, (Ull)b[CHIP], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);

                                spmm_core1_load      ( 5,  4,  3,  1,     r1,      0);
                                spmm_core1           ( 6,  5,             r1);
                                spmm_core1_load      ( 7,  6,  5,  2,     r2,      0);
                                spmm_core1           ( 8,  7,             r2);
                                spmm_core1_load      ( 9,  8,  7,  3,     r3,      0);
                                spmm_core1           (10,  9,             r3);
                                spmm_core1_end       (11, 10,  9,  4,  5,  6,  7,  0);

                                spmm_core1_start     (14, 13, 12, 11,  4, r0,      0);
                                spmm_core1_load      (15, 14, 13,  5,     r1,      0);
                                spmm_core1           (16, 15,             r1);
                                spmm_core1_load      (17, 16, 15,  6,     r2,      0);
                                spmm_core1           (18, 17,             r2);
                                spmm_core1_load      (19, 18, 17,  7,     r3,      0);
                                spmm_core1           (20, 19,             r3);
                                spmm_core1_end       (21, 20, 19,  8,  9, 10, 11,  8);

                                spmm_core1_start     (24, 23, 22, 21,  8, r0,      8);
                                spmm_core1_load      (25, 24, 23,      9, r1,      8);
                                spmm_core1           (26, 25,             r1);
                                spmm_core1_load      (27, 26, 25,     10, r2,      8);
                                spmm_core1           (28, 27,             r2);
                                spmm_core1_load      (29, 28, 27,     11, r3,      8);
                                spmm_core1           (30, 29,             r3);
                                spmm_core1_end       (31, 30, 29, 12, 13, 14, 15,  8);

                                spmm_core1_start     (34, 33, 32, 31, 12, r0,      8);
                                spmm_core1_load      (35, 34, 33,     13, r1,      8);
                                spmm_core1           (36, 35,             r1);
                                spmm_core1_load      (37, 36, 35,     14, r2,      8);
                                spmm_core1           (38, 37,             r2);
                                spmm_core1_load      (39, 38, 37,     15, r3,      8);
                                spmm_core1           (40, 39,             r3);
                                spmm_core1_end       (41, 40, 39, 16, 17, 18, 19, 16);

                                spmm_core1_start     (44, 43, 42, 41, 16, r0,     16);
                                spmm_core1_load      (45, 44, 43,     17, r1,     16);
                                spmm_core1           (46, 45,             r1);
                                spmm_core1_load      (47, 46, 45,     18, r2,     16);
                                spmm_core1           (48, 47,             r2);
                                spmm_core1_load      (49, 48, 47,     19, r3,     16);
                                spmm_core1           (50, 49,             r3);
                                spmm_core1_last_end  (51, 50, 49, 20, 21, 22,     16);

                                spmm_core1_last_start(54, 53, 52, 51, 20, r0,     16);
                                spmm_core1_load      (55, 54, 53,     21, r1,     16);
                                spmm_core1           (56, 55,             r1);
                                spmm_core1_load      (57, 56, 55,     22, r2,     16);
                                spmm_core1           (58, 57,             r2);

                                exe(OP_FMA, &AR[59][0], AR[58][0], EXP_H3210, BR[57][2][1], EXP_H3232, BR[58][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                                exe(OP_FMA, &AR[59][1], AR[58][1], EXP_H3210, BR[57][2][1], EXP_H3232, BR[58][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                                exe(OP_FMA, &AR[59][2], AR[58][2], EXP_H3210, BR[57][2][1], EXP_H3232, BR[58][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                                exe(OP_FMA, &AR[59][3], AR[58][3], EXP_H3210, BR[57][2][1], EXP_H3232, BR[58][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                                mop(OP_LDWR, 1, &BR[59][0][1], (Ull)a_row_index, (Ull)rofs, MSK_W0, (Ull)a_row_index, A_row_blk_size, 0, 0, (Ull)NULL, A_row_blk_size);

                                exe(OP_ADD, &r0, BR[59][0][1], EXP_H3210, oofs, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffLL, OP_NOP, 0LL);
                                spmm_final(62, 59, r0);
                            }
                        }
                    }
//EMAX5A end
                }
            }
        }
    }
//EMAX5A drain_dirty_lmm
}

void mm(IMAXDenseMatrix *result, IMAXDenseMatrix *imax_a, IMAXDenseMatrix *imax_b) {
    Ull CHIP;
    Ull LOOP1, LOOP0;
    Ull INIT1, INIT0;
    Ull blk, end_sum, nnz_col_blk, a_row_blk, b_col_blk;
    Ull AR[64][4];    /* output of EX in each unit */
    Ull BR[64][4][4]; /* output registers in each unit */
    Ull r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
    Ull r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
    Ull x0, x1, x2, x3;
    Ull cc0, cc1, cc2, cc3, ex0, ex1;
    Ull cofs, rofs, oofs, k;
    Ull cofs1;

    Uint W = 4;
    Ull A_row_blk_size = imax_a->row_blk_size;
    Ull A_col_blk_size = imax_a->col_blk_size;
    Ull A_row_size = imax_a->row_padded_size;
    Ull A_col_size = imax_a->col_padded_size;
    Ull B_row_size = imax_b->col_padded_size;
    Ull B_col_size = imax_b->col_padded_size;
    Ull B_row_blk_size = imax_b->row_blk_size;
    Ull B_col_blk_size = imax_b->col_blk_size;

    Ull cofs_init = (0-W*4*2*A_row_blk_size)<<32|((0-W*4*2*B_row_blk_size)&0xffffffff);
    Ull rofs_init = (0-1*8LL)<<32|((0-1*4LL)&0xffffffff);

    Ull A_blk_size = A_row_blk_size * A_col_blk_size;
    Ull B_blk_size = B_row_blk_size * B_col_blk_size;
    Ull C_blk_size = A_row_blk_size * B_col_blk_size;
    Ull A_row_blk_size_mul_2 = A_row_blk_size*2;
    Ull A_row_blk_size_mul_8_2 = A_row_blk_size*8*2;
    Ull A_row_size_mul_W_4_2 = (W*4*2*B_row_blk_size)<<32|((W*4*2*B_row_blk_size)&0xffffffff);

    #define H 57
    Uint *a[H][NCHIP], *a0[NCHIP];
    Uint *b[H], *b0[H], *b1[H], *b2[H], *b3[H];
    Uint *c0[H], *c00[H], *c01[H], *c02[H], *c03[H];
    Uint *a_row_index;
    Ull A_nnz_size = 0;
    Ull a_col_blk = 0;
    Ull a_col_blk_iter = 0;
    Uint *a_head = (Uint*)imax_a->val;
    Uint *b_head = (Uint*)imax_b->val;
    Uint *c_head = (Uint*)result->val;

    printf("<<IMAX>>\n");

    // Select Column of A(=Row of B)
    for (a_col_blk = 0, a_col_blk_iter = 0; a_col_blk < A_col_size; a_col_blk += A_col_blk_size, a_col_blk_iter += 1) {
        // Select Row of A(=Row of C)
        for (a_row_blk = 0, end_sum = 0; a_row_blk < A_row_size; a_row_blk += A_row_blk_size, end_sum += A_nnz_size * A_row_blk_size) { // A_row_blk
            // Select Column of B(= Column of C)
            for (b_col_blk = 0; b_col_blk < B_col_size; b_col_blk += B_col_blk_size) {
                for (k = 0; k < H; k++) {
                    for (CHIP = 0; CHIP < NCHIP; CHIP++) {
                        a0[CHIP] = (Uint*)a_head + a_row_blk * A_col_size + (CHIP * A_col_size / NCHIP + a_col_blk) * A_row_blk_size;
                        a[k][CHIP] = (Uint*)a0[CHIP] + (2 * k * A_row_blk_size);
                    }

                    b[k] = (Uint*)b_head + a_col_blk * B_col_size + (k * B_col_size / H + b_col_blk) * B_row_blk_size;
                    b0[k] = (Uint*)b[k];
                    b1[k] = (Uint*)b[k] + B_row_blk_size * 2;
                    b2[k] = (Uint*)b[k] + B_row_blk_size * 4;
                    b3[k] = (Uint*)b[k] + B_row_blk_size * 6;

                    c0[k] = (Uint*)c_head + a_row_blk * B_col_size + (k * B_col_size / H + b_col_blk) * A_row_blk_size;
                    c00[k] = (Uint*)c0[k];
                    c01[k] = (Uint*)c0[k] + A_row_blk_size * 2;
                    c02[k] = (Uint*)c0[k] + A_row_blk_size * 4;
                    c03[k] = (Uint*)c0[k] + A_row_blk_size * 6;
                }
                #define sgemm_core1_0(r, rm1, index) \
                            mop(OP_LDR,  3, &BR[rm1][0][1], (Ull)b0[index],      (Ull)cofs, MSK_W1, (Ull)b[index], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);  \
                            mop(OP_LDR,  3, &BR[rm1][0][0], (Ull)b1[index],      (Ull)cofs, MSK_W1, (Ull)b[index], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);  \
                            mop(OP_LDR,  3, &BR[rm1][1][1], (Ull)b2[index],      (Ull)cofs, MSK_W1, (Ull)b[index], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);  \
                            mop(OP_LDR,  3, &BR[rm1][1][0], (Ull)b3[index],      (Ull)cofs, MSK_W1, (Ull)b[index], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);  \
                            mop(OP_LDWR, 1, &BR[rm1][2][1], (Ull)a[index][CHIP], (Ull)rofs, MSK_W0, (Ull)a0[CHIP], A_blk_size, 0, 0, (Ull)NULL, A_blk_size);  \
                            exe(OP_FMA, &AR[r][0], AR[rm1][0], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      \
                            exe(OP_FMA, &AR[r][1], AR[rm1][1], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      \
                            exe(OP_FMA, &AR[r][2], AR[rm1][2], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      \
                            exe(OP_FMA, &AR[r][3], AR[rm1][3], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)
                
                #define sgemm_final(r, rm1) \
                            mop(OP_LDR, 3, &BR[r][0][1], (Ull)c00[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], C_blk_size, 0, 1, (Ull)NULL, C_blk_size);  \
                            mop(OP_LDR, 3, &BR[r][1][1], (Ull)c01[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], C_blk_size, 0, 1, (Ull)NULL, C_blk_size);  \
                            mop(OP_LDR, 3, &BR[r][2][1], (Ull)c02[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], C_blk_size, 0, 1, (Ull)NULL, C_blk_size);  \
                            mop(OP_LDR, 3, &BR[r][3][1], (Ull)c03[CHIP], (Ull)oofs, MSK_W0, (Ull)c0[CHIP], C_blk_size, 0, 1, (Ull)NULL, C_blk_size);  \
                            exe(OP_FAD, &AR[r][0], AR[rm1][0], EXP_H3210, BR[r][0][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);          \
                            exe(OP_FAD, &AR[r][1], AR[rm1][1], EXP_H3210, BR[r][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);          \
                            exe(OP_FAD, &AR[r][2], AR[rm1][2], EXP_H3210, BR[r][2][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);          \
                            exe(OP_FAD, &AR[r][3], AR[rm1][3], EXP_H3210, BR[r][3][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);          \
                            mop(OP_STR, 3, &AR[r][0], (Ull)oofs, (Ull)c00[CHIP], MSK_D0, (Ull)c0[CHIP], C_blk_size, 0, 1, (Ull)NULL, C_blk_size);     \
                            mop(OP_STR, 3, &AR[r][1], (Ull)oofs, (Ull)c01[CHIP], MSK_D0, (Ull)c0[CHIP], C_blk_size, 0, 1, (Ull)NULL, C_blk_size);     \
                            mop(OP_STR, 3, &AR[r][2], (Ull)oofs, (Ull)c02[CHIP], MSK_D0, (Ull)c0[CHIP], C_blk_size, 0, 1, (Ull)NULL, C_blk_size);     \
                            mop(OP_STR, 3, &AR[r][3], (Ull)oofs, (Ull)c03[CHIP], MSK_D0, (Ull)c0[CHIP], C_blk_size, 0, 1, (Ull)NULL, C_blk_size)
            
//EMAX5A begin sgemm1 mapdist=0
                for (CHIP=0;CHIP<NCHIP;CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
                    for (INIT1=1,LOOP1=B_col_blk_size/(W*2),cofs=cofs_init;LOOP1--;INIT1=0) {
                        for (INIT0=1,LOOP0=A_row_blk_size,rofs=rofs_init;LOOP0--;INIT0=0) {
                            exe(OP_ADD, &cofs,            cofs, EXP_H3210,     INIT0?A_row_size_mul_W_4_2:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP,                  0LL, OP_NOP, 0LL);
                            exe(OP_ADD, &rofs, INIT0?rofs:rofs, EXP_H3210, (1*8LL)<<32|((1*4LL)&0xffffffff), EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL);
                            exe(OP_ADD, &oofs,            cofs, EXP_H3232, 0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);

                            mop(OP_LDR,  3, &BR[2][0][1], (Ull)b0[0],      (Ull)cofs, MSK_W1, (Ull)b[0], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);
                            mop(OP_LDR,  3, &BR[2][0][0], (Ull)b1[0],      (Ull)cofs, MSK_W1, (Ull)b[0], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);
                            mop(OP_LDR,  3, &BR[2][1][1], (Ull)b2[0],      (Ull)cofs, MSK_W1, (Ull)b[0], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);
                            mop(OP_LDR,  3, &BR[2][1][0], (Ull)b3[0],      (Ull)cofs, MSK_W1, (Ull)b[0], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);
                            mop(OP_LDWR, 1, &BR[2][2][1], (Ull)a[0][CHIP], (Ull)rofs, MSK_W0, (Ull)a0[CHIP], A_blk_size, 0, 0, (Ull)NULL, A_blk_size);
                            exe(OP_FML, &AR[3][0], BR[2][0][1], EXP_H3210, BR[2][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                            exe(OP_FML, &AR[3][1], BR[2][0][0], EXP_H3210, BR[2][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                            exe(OP_FML, &AR[3][2], BR[2][1][1], EXP_H3210, BR[2][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                            exe(OP_FML, &AR[3][3], BR[2][1][0], EXP_H3210, BR[2][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);

                            sgemm_core1_0( 4,  3,  1);
                            sgemm_core1_0( 5,  4,  2);
                            sgemm_core1_0( 6,  5,  3);
                            sgemm_core1_0( 7,  6,  4);
                            sgemm_core1_0( 8,  7,  5);
                            sgemm_core1_0( 9,  8,  6);

                            sgemm_core1_0(10,  9,  7);
                            sgemm_core1_0(11, 10,  8);
                            sgemm_core1_0(12, 11,  9);
                            sgemm_core1_0(13, 12, 10);
                            sgemm_core1_0(14, 13, 11);
                            sgemm_core1_0(15, 14, 12);
                            sgemm_core1_0(16, 15, 13);
                            sgemm_core1_0(17, 16, 14);
                            sgemm_core1_0(18, 17, 15);
                            sgemm_core1_0(19, 18, 16);
                            sgemm_core1_0(20, 19, 17);

                            sgemm_core1_0(21, 20, 18);
                            sgemm_core1_0(22, 21, 19);
                            sgemm_core1_0(23, 22, 20);
                            sgemm_core1_0(24, 23, 21);
                            sgemm_core1_0(25, 24, 22);
                            sgemm_core1_0(26, 25, 23);
                            sgemm_core1_0(27, 26, 24);
                            sgemm_core1_0(28, 27, 25);
                            sgemm_core1_0(29, 28, 26);
                            sgemm_core1_0(30, 29, 27);

                            sgemm_core1_0(31, 30, 28);
                            sgemm_core1_0(32, 31, 29);
                            sgemm_core1_0(33, 32, 30);
                            sgemm_core1_0(34, 33, 31);
                            sgemm_core1_0(35, 34, 32);
                            sgemm_core1_0(36, 35, 33);
                            sgemm_core1_0(37, 36, 34);
                            sgemm_core1_0(38, 37, 35);
                            sgemm_core1_0(39, 38, 36);

                            sgemm_core1_0(40, 39, 37);
                            sgemm_core1_0(41, 40, 38);
                            sgemm_core1_0(42, 41, 39);
                            sgemm_core1_0(43, 42, 40);
                            sgemm_core1_0(44, 43, 41);
                            sgemm_core1_0(45, 44, 42);
                            sgemm_core1_0(46, 45, 43);
                            sgemm_core1_0(47, 46, 44);
                            sgemm_core1_0(48, 47, 45);
                            sgemm_core1_0(49, 48, 46);

                            sgemm_core1_0(50, 49, 47);
                            sgemm_core1_0(51, 50, 48);
                            sgemm_core1_0(52, 51, 49);
                            sgemm_core1_0(53, 52, 50);
                            sgemm_core1_0(54, 53, 51);
                            sgemm_core1_0(55, 54, 52);
                            sgemm_core1_0(56, 55, 53);
                            sgemm_core1_0(57, 56, 54);
                            sgemm_core1_0(58, 57, 55);
                            sgemm_core1_0(59, 58, 56);

                            sgemm_final(62, 59);
                        }
                    }
                }
//EMAX5A end
            }
        }
    }
//EMAX5A drain_dirty_lmm
}

void relu(DenseMatrix *result, DenseMatrix *a) {
    for (int i = 0; i < (a->row_size * a->col_size); i++) {
        result->val[i] = (a->val[i] > 0) ? a->val[i] : 0;
    }
}

#endif