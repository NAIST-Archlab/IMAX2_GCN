// EMAX6/7 GCN Test Program            //
// linalg_imax.c                       //
//         Copyright (C) 2024 by NAIST //
//          Primary writer: Dohyun Kim //
//          kim.dohyun.kg7@is.naist.jp //
#if defined(EMAX6) || defined(EMAX7)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/linalg.h"
#if defined(EMAX7)
#include "../conv-c2d/emax7lib.c"
#elif defined(EMAX6)
#include "../conv-c2d/emax6lib.c"
#endif

Uchar *membase = NULL;
Uchar *prev = NULL;

Uchar* sysinit(Uint memsize, Uint alignment) {
#if defined(ARMZYNQ) && defined(EMAX6)
  if (emax6_open() == NULL) 
    exit(1);
  membase = emax_info.ddr_mmap;
  {int i; for (i=0; i<(memsize+sizeof(Dll)-1)/sizeof(Dll); i++) *((Dll*)membase+i)=0;}
  prev = (Uchar*)((Dll*)membase + (memsize+sizeof(Dll)-1)/sizeof(Dll));
#elif defined(ARMZYNQ) && defined(EMAX7)
  if (emax7_open(1) == NULL) 
    exit(1);
  membase = emax_info[0].ddr_mmap;
  {int i; for (i=0; i<(memsize+sizeof(Dll)-1)/sizeof(Dll); i++) *((Dll*)membase+i)=0;}
  prev = (Uchar*)((Dll*)membase + (memsize+sizeof(Dll)-1)/sizeof(Dll));
#elif __linux__ == 1
  posix_memalign(&membase, alignment, memsize);
#else
  if (membase != NULL) {return membase;}
  membase = (void*)malloc(memsize+alignment);
  prev = membase;
  if ((Ull)membase & (Ull)(alignment-1)) {membase = (void*)(((Ull)membase & ~(Ull)(alignment-1))+alignment);}
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
#if (defined(ARMSIML) || defined(ARMZYNQ)) && defined(EMAX7)
  emax7[0].dma_ctrl  = emax_info[0].dma_mmap;
  emax7[0].reg_ctrl  = emax_info[0].reg_mmap;
  ((struct reg_ctrl*)emax7[0].reg_ctrl)->i[0].cmd = CMD_RESET;
#if defined(ARMZYNQ)
  usleep(1);
#endif
  switch (((struct reg_ctrl*)emax7[0].reg_ctrl)->i[0].stat>>8 & 0xf) {
  case  3:EMAX_DEPTH = 64;break;
  case  2:EMAX_DEPTH = 32;break;
  case  1:EMAX_DEPTH = 16;break;
  default:EMAX_DEPTH =  8;break;
  }
  ((struct reg_ctrl*)emax7[0].reg_ctrl)->i[0].adtr = emax_info[0].ddr_mmap - emax_info[0].lmm_phys;
  ((struct reg_ctrl*)emax7[0].reg_ctrl)->i[0].dmrp = 0LL;
#endif
  return membase;
}

Uchar* imax_alloc(Uint memsize, Uint alignment) {
    if (membase == NULL) {return sysinit(memsize, alignment);}
    else {
        #if defined(ARMZYNQ) && (defined(EMAX6) || defined(EMAX7))
        {int i; for (i=0; i<(memsize+sizeof(Dll)-1)/sizeof(Dll); i++) *((Dll*)prev+i)=0;}
        prev = (Dll*)prev + ((memsize+sizeof(Dll)-1)/sizeof(Dll));
        #elif __linux__ == 1
        posix_memalign(&membase, alignment, memsize);
        #else
        if (membase != NULL) {return membase;}
        membase = (void*)malloc(memsize+alignment);
        prev = membase;
        if ((Ull)membase & (Ull)(alignment-1)) {membase = (void*)(((Ull)membase & ~(Ull)(alignment-1))+alignment);}
        #endif
        return prev;
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
    Ull  l[2];
    Dll  d;
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
      *(Ull*)dst = buf.l[0];
    }
    else {
      *(Ull*)dst = *(Ull*)src;
      src += sizeof(Ull)/sizeof(Uint);
    }
    dst += sizeof(Ull)/sizeof(Uint);
    words-=2;
  }

  if (loop = words/(sizeof(Dll)/sizeof(Uint))) {
    if ((Ull)src & sizeof(Uint)) {
      for(i=0; i<loop; i++) {
	buf.i[0] = *src++;
	buf.i[1] = *src++;
	buf.i[2] = *src++;
	buf.i[3] = *src++;
	*(Dll*)dst = buf.d;
	dst += sizeof(Dll)/sizeof(Uint);
      }
    }
    else if ((Ull)src & sizeof(Ull)) {
      for(i=0; i<loop; i++) {
	buf.l[0] = *(Ull*)src;src += sizeof(Ull)/sizeof(Uint);
	buf.l[1] = *(Ull*)src;src += sizeof(Ull)/sizeof(Uint);
	*(Dll*)dst = buf.d;
	dst += sizeof(Dll)/sizeof(Uint);
      }
    }
    else {
      for(i=0; i<loop; i++) {
	*(Dll*)dst = *(Dll*)src;
	src += sizeof(Dll)/sizeof(Uint);
	dst += sizeof(Dll)/sizeof(Uint);
      }
    }
    words -= loop*(sizeof(Dll)/sizeof(Uint));
  }

  if (words >= 2) { /* 8B-access */
    if ((Ull)src & sizeof(Uint)) {
      buf.i[0] = *src++;
      buf.i[1] = *src++;
      *(Ull*)dst = buf.l[0];
    }
    else {
      *(Ull*)dst = *(Ull*)src;
      src += sizeof(Ull)/sizeof(Uint);
    }
    dst += sizeof(Ull)/sizeof(Uint);
    words-=2;
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
    *(Ull*)dst = 0;
    dst += sizeof(Ull)/sizeof(Uint);
    words-=2;
  }

  if (loop = words/(sizeof(Dll)/sizeof(Uint))) {
    for(i=0; i<loop; i++) {
#if __AARCH64EL__ == 1
      *((Dll*)dst) = 0;
#else
      ((Dll*)dst)->u[0] = 0;
      ((Dll*)dst)->u[1] = 0;
#endif
      dst += sizeof(Dll)/sizeof(Uint);
    }
    words -= loop*(sizeof(Dll)/sizeof(Uint));
  }

  if (words >= 2) { /* 8B-access */
    *(Ull*)dst = 0;
    dst += sizeof(Ull)/sizeof(Uint);
    words-=2;
  }
  if (words >= 1) { /* 4B-access */
    *dst++ = 0;
    words--;
  }
}

void mm(IMAXDenseMatrix *result, IMAXDenseMatrix *imax_a, IMAXDenseMatrix *imax_b) {
    Ull CHIP;
    Ull LOOP1, LOOP0;
    Ull INIT1, INIT0;
    Ull blk, end_sum, nnz_col_blk, a_row_blk, b_col_blk, b_row, b_row_iter;
    Ull AR[64][4];    /* output of EX in each unit */
    Ull BR[64][4][4]; /* output registers in each unit */
    Ull r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
    Ull r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
    Ull x0, x1, x2, x3;
    Ull cc0, cc1, cc2, cc3, ex0, ex1;
    Ull cofs, rofs, oofs, k;

    Uint W = 4;
    Ull A_row_size = imax_a->row_padded_size;
    Ull A_col_size = imax_a->col_padded_size;
    Ull B_row_size = imax_b->row_padded_size;
    Ull B_col_size = imax_b->col_padded_size;
    Ull C_row_size = result->row_padded_size;
    Ull C_col_size = result->col_padded_size;
    Ull A_blk_row_size = imax_a->blk_row_size;
    Ull A_blk_col_size = imax_a->blk_col_size;
    Ull B_blk_row_size = imax_b->blk_row_size;
    Ull B_blk_col_size = imax_b->blk_col_size;
    Ull C_blk_row_size = result->blk_row_size;
    Ull C_blk_col_size = result->blk_col_size;
    //if (A_blk_col_size != B_blk_row_size) return;
    //else if (B_blk_col_size != C_blk_col_size) return;
    //else if (A_blk_row_size != C_blk_row_size) return;
    //else if (A_col_size != B_row_size) return;
    if (A_col_size != B_row_size) return;
    else if (B_col_size != C_col_size) return;
    else if (A_row_size != C_row_size) return;

    Ull cofs_init = (0-W*4*2*B_blk_row_size)<<32|((0-W*4*2*C_blk_row_size)&0xffffffff);
    Ull rofs_init = (0-1*8LL)<<32|((0-1*8LL)&0xffffffff);

    Ull A_blk_size = A_blk_row_size * A_blk_col_size;
    Ull B_blk_size = B_blk_row_size * B_blk_col_size;
    Ull C_blk_size = A_blk_row_size * B_blk_col_size;
    Ull BC_W_blk_size = (W*4*2*B_blk_row_size)<<32|((W*4*2*C_blk_row_size)&0xffffffff);

    Uint *a[MM_H/2][NCHIP], *a0[NCHIP];
    Uint *b, *b0[MM_H], *b1[MM_H], *b2[MM_H], *b3[MM_H];
    Uint *c0[NCHIP], *c00[NCHIP], *c01[NCHIP], *c02[NCHIP], *c03[NCHIP];
    Ull a_col_blk = 0;
    Uint *a_head = (Uint*)imax_a->val;
    Uint *b_head = (Uint*)imax_b->val;
    Uint *c_head = (Uint*)result->val;

    printf("<<IMAX>>\n");
    printf("Dense Input  Head: %08x_%08x\n", (Uint)((Ull)a_head >> 32), (Uint)a_head);
    printf("Dense Input  Head: %08x_%08x\n", (Uint)((Ull)b_head >> 32), (Uint)b_head);
    printf("Dense Input  Head: %08x_%08x\n", (Uint)((Ull)c_head >> 32), (Uint)c_head);
    printf("Orig(%d,%d)x(%d,%d)=(%d,%d) Padded(%d,%d)x(%d,%d)=(%d,%d) blk(%d,%d)(%d,%d)(%d,%d)\n", 
        imax_a->row_size, imax_a->col_size, imax_b->row_size, imax_b->col_size, result->row_size, result->col_size,
        imax_a->row_padded_size, imax_a->col_padded_size, imax_b->row_padded_size, imax_b->col_padded_size, result->row_padded_size, result->col_padded_size,
        imax_a->blk_row_size, imax_a->blk_col_size, imax_b->blk_row_size, imax_b->blk_col_size,result->blk_row_size, result->blk_col_size
    );
    #ifdef EMAX7
        int LANE = 0;
        reset_nanosec(0);
    #else
        reset_nanosec();
    #endif
    // Select Row of A(=Row of C)
    for (a_row_blk = 0; a_row_blk < A_row_size/NCHIP; a_row_blk += A_blk_row_size) {
        // Select Column of B(= Column of C)
        for (b_col_blk = 0; b_col_blk < B_col_size; b_col_blk += B_blk_col_size) {
            // Select Column of A(=Row of B)
            for (a_col_blk = 0; a_col_blk < A_col_size; a_col_blk += B_blk_row_size) {
                for (CHIP = 0; CHIP < NCHIP; CHIP++) {
                    // TODO: マルチCHIP対応
                    // 現状だとうまく動かない
                    a0[CHIP] = (Uint*)a_head + ((CHIP * A_row_size)/NCHIP + a_row_blk)*A_col_size + a_col_blk*A_blk_row_size;
                    c0[CHIP] = (Uint*)c_head + ((CHIP * C_row_size)/NCHIP + a_row_blk)*C_col_size + b_col_blk*C_blk_row_size;
                    c00[CHIP] = (Uint*)c0[CHIP];
                    c01[CHIP] = (Uint*)c0[CHIP] + C_blk_row_size * 2;
                    c02[CHIP] = (Uint*)c0[CHIP] + C_blk_row_size * 4;
                    c03[CHIP] = (Uint*)c0[CHIP] + C_blk_row_size * 6;
                }
                for (b_row = 0; b_row < B_blk_row_size; b_row += MM_H) {
                    b = (Uint*)b_head + (a_col_blk*B_col_size) + (b_col_blk*B_blk_row_size) + 2*b_row;
                    for (k = 0; k < MM_H; k++) {
                        for (CHIP = 0; CHIP < NCHIP; CHIP++) {
                            if (k%2 == 0) a[k/2][CHIP] = (Uint*)a0[CHIP] + (2*A_blk_row_size)*((b_row+k)/2);
                        }
                        b0[k] = (Uint*)b + 2*k;
                        b1[k] = (Uint*)b0[k] + B_blk_row_size * 2;
                        b2[k] = (Uint*)b0[k] + B_blk_row_size * 4;
                        b3[k] = (Uint*)b0[k] + B_blk_row_size * 6;
                    }

                    #define sgemm_core1_0(r, rm1, index, a_index) \
                                mop(OP_LDR,  3, &BR[rm1][0][1], (Ull)b0[index],      (Ull)cofs, MSK_W1, (Ull)b0[index], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);  \
                                mop(OP_LDR,  3, &BR[rm1][0][0], (Ull)b1[index],      (Ull)cofs, MSK_W1, (Ull)b0[index], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);  \
                                mop(OP_LDR,  3, &BR[rm1][1][1], (Ull)b2[index],      (Ull)cofs, MSK_W1, (Ull)b0[index], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);  \
                                mop(OP_LDR,  3, &BR[rm1][1][0], (Ull)b3[index],      (Ull)cofs, MSK_W1, (Ull)b0[index], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);  \
                                mop(OP_LDR,  3, &BR[rm1][2][1], (Ull)a[a_index][CHIP], (Ull)rofs, MSK_W1, (Ull)a[a_index][CHIP], A_blk_size, 0, 0, (Ull)NULL, A_blk_size);  \
                                exe(OP_FMA, &AR[r][0], AR[rm1][0], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      \
                                exe(OP_FMA, &AR[r][1], AR[rm1][1], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      \
                                exe(OP_FMA, &AR[r][2], AR[rm1][2], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      \
                                exe(OP_FMA, &AR[r][3], AR[rm1][3], EXP_H3210, BR[rm1][2][1], EXP_H1010, BR[rm1][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)

                    #define sgemm_core1_1(r, rm1, rm2, index) \
                                mop(OP_LDR,  3, &BR[rm1][0][1], (Ull)b0[index],      (Ull)cofs, MSK_W1, (Ull)b0[index], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);  \
                                mop(OP_LDR,  3, &BR[rm1][0][0], (Ull)b1[index],      (Ull)cofs, MSK_W1, (Ull)b0[index], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);  \
                                mop(OP_LDR,  3, &BR[rm1][1][1], (Ull)b2[index],      (Ull)cofs, MSK_W1, (Ull)b0[index], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);  \
                                mop(OP_LDR,  3, &BR[rm1][1][0], (Ull)b3[index],      (Ull)cofs, MSK_W1, (Ull)b0[index], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);  \
                                exe(OP_FMA, &AR[r][0], AR[rm1][0], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      \
                                exe(OP_FMA, &AR[r][1], AR[rm1][1], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      \
                                exe(OP_FMA, &AR[r][2], AR[rm1][2], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      \
                                exe(OP_FMA, &AR[r][3], AR[rm1][3], EXP_H3210, BR[rm2][2][1], EXP_H3232, BR[rm1][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL)
                    
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
                    for (CHIP=0;CHIP<NCHIP;CHIP++) {
                        for (INIT1=1,LOOP1=A_blk_row_size,rofs=rofs_init;LOOP1--;INIT1=0) {
                            for (INIT0=1,LOOP0=B_blk_col_size/(W*2),cofs=cofs_init;LOOP0--;INIT0=0) {
                                exe(OP_ADD, &cofs, INIT0?cofs:cofs, EXP_H3210, BC_W_blk_size, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL);
                                exe(OP_ADD, &rofs, rofs, EXP_H3210, INIT0?(1*8LL)<<32|((1*8LL)&0xffffffff):0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                                exe(OP_ADD, &oofs, rofs, EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffLL, OP_NOP, 0LL);

                                mop(OP_LDR,  3, &BR[1][0][1], (Ull)b0[0],      (Ull)cofs, MSK_W1, (Ull)b0[0], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);
                                mop(OP_LDR,  3, &BR[1][0][0], (Ull)b1[0],      (Ull)cofs, MSK_W1, (Ull)b0[0], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);
                                mop(OP_LDR,  3, &BR[1][1][1], (Ull)b2[0],      (Ull)cofs, MSK_W1, (Ull)b0[0], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);
                                mop(OP_LDR,  3, &BR[1][1][0], (Ull)b3[0],      (Ull)cofs, MSK_W1, (Ull)b0[0], B_blk_size, 0, 0, (Ull)NULL, B_blk_size);
                                mop(OP_LDR,  3, &BR[1][2][1], (Ull)a[0][CHIP], (Ull)rofs, MSK_W1, (Ull)a[0][CHIP], A_blk_size, 0, 0, (Ull)NULL, A_blk_size);
                                exe(OP_FML, &AR[2][0], BR[1][0][1], EXP_H3210, BR[1][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                                exe(OP_FML, &AR[2][1], BR[1][0][0], EXP_H3210, BR[1][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                                exe(OP_FML, &AR[2][2], BR[1][1][1], EXP_H3210, BR[1][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                                exe(OP_FML, &AR[2][3], BR[1][1][0], EXP_H3210, BR[1][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);

                                sgemm_core1_1( 3,  2,  1,  1);
                                sgemm_core1_0( 4,  3,  2,  1);
                                sgemm_core1_1( 5,  4,  3,  3);
                                sgemm_core1_0( 6,  5,  4,  2);
                                sgemm_core1_1( 7,  6,  5,  5);
                                sgemm_core1_0( 8,  7,  6,  3);

                                sgemm_core1_1( 9,  8,  7,  7);
                                sgemm_core1_0(10,  9,  8,  4);
                                sgemm_core1_1(11, 10,  9,  9);
                                sgemm_core1_0(12, 11, 10,  5);
                                sgemm_core1_1(13, 12, 11, 11);
                                sgemm_core1_0(14, 13, 12,  6);
                                sgemm_core1_1(15, 14, 13, 13);
                                sgemm_core1_0(16, 15, 14,  7);
                                sgemm_core1_1(17, 16, 15, 15);

#ifdef HARD_UNIT32 
                                sgemm_final(30, 17);
#else
                                sgemm_core1_0(18, 17, 16, 8);
                                sgemm_core1_1(19, 18, 17, 17);

                                sgemm_core1_0(20, 19, 18, 9);
                                sgemm_core1_1(21, 20, 19, 19);
                                sgemm_core1_0(22, 21, 20, 10);
                                sgemm_core1_1(23, 22, 21, 21);
                                sgemm_core1_0(24, 23, 22, 11);
                                sgemm_core1_1(25, 24, 23, 23);
                                sgemm_core1_0(26, 25, 24, 12);
                                sgemm_core1_1(27, 26, 25, 25);
                                sgemm_core1_0(28, 27, 26, 13);
                                sgemm_core1_1(29, 28, 27, 27);

                                sgemm_core1_0(30, 29, 28, 14);
                                sgemm_core1_1(31, 30, 29, 29);
                                sgemm_core1_0(32, 31, 30, 15);
                                sgemm_core1_1(33, 32, 31, 31);

                                sgemm_final(62, 33);
#endif
                            }
                        }
                    }
//EMAX5A end
                }
            }
        }
    }
//EMAX5A drain_dirty_lmm
    #ifdef EMAX7
        get_nanosec(0, 0);
        for (int i = 0; i < 8; i++) all_nanosec[MM][i] += nanosec[0][i];
        show_nanosec(0);
    #else
        get_nanosec(0);
        for (int i = 0; i < 8; i++) all_nanosec[MM][i] += nanosec[i];
        show_nanosec();
    #endif
}

void relu(DenseMatrix *result, DenseMatrix *a) {
    for (int i = 0; i < (a->row_size * a->col_size); i++) {
        result->val[i] = (a->val[i] > 0) ? a->val[i] : 0;
    }
}

void d_relu(DenseMatrix *result, DenseMatrix *a) {
    for (int i = 0; i < (a->col_size * a->row_size); i++) {
        result->val[i] = (a->val[i] > 0) ? a->val[i] : 0;
    }
}

#endif