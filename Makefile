## EMAX6/7 GCN Test Program            ##
##         Copyright (C) 2023 by NAIST ##
##          Primary writer: Dohyun Kim ##
##          kim.dohyun.kg7@is.naist.jp ##
PROGRAM := imax_gcn
TEST_SPARSE_PROGRAM := test_sparse
TEST_DENSE_PROGRAM := test_dense
SRC_DIR := src
TEST_DIR := test
INCLUDE := ./include/
CONV := ./conv-c2c/
SRCS := $(wildcard $(SRC_DIR)/*.c)
TEST_SRCS := $(wildcard $(TEST_DIR)/*.c)
OBJS := $(SRCS:.c=.o)
MAIN := main.c
MAIN_OBJS := main.o
NCHIP := 1
CPUONLY := 0
CUDA := 0
SAME_DISTANCE := 0
UNIT32 := 1
HARD_UNIT32 := 0
LMM128 := 1
EMAX_VER := 7
EMAX_DEFINE := -DEMAX6 -DDEBUG -DUSE_MP -DNCHIP=$(NCHIP)
TEST_SPARSE_OBJS := test/test_sparse.o
TEST_DENSE_OBJS := test/test_dense.o
HEADERS := $(CONV)/emax6.h $(INCLUDE)/layer.h $(INCLUDE)/options.h $(INCLUDE)/sparse.h $(INCLUDE)/utils.h $(INCLUDE)/reader.h $(INCLUDE)/gcn.h $(INCLUDE)/optimizer.h
ifeq ($(EMAX_VER), 7)
CONV := ./conv-c2d
HEADERS := $(CONV)/emax7.h $(INCLUDE)/layer.h $(INCLUDE)/options.h $(INCLUDE)/sparse.h $(INCLUDE)/utils.h $(INCLUDE)/reader.h $(INCLUDE)/gcn.h  $(INCLUDE)/optimizer.h
EMAX_DEFINE := -DCBLAS_GEMM -DEMAX7 -DDEBUG -DUSE_MP -DNCHIP=$(NCHIP)
endif
ifeq ($(UNIT32), 1)
EMAX_DEFINE := $(EMAX_DEFINE) -DUNIT32
endif
ifeq ($(HARD_UNIT32), 1)
EMAX_DEFINE := $(EMAX_DEFINE) -DHARD_UNIT32
endif
ifeq ($(LMM128), 1)
EMAX_DEFINE := $(EMAX_DEFINE) -DLMM128
endif

ifeq ($(MACHTYPE),x86_64)
	X64 ?= 1
endif

ARM ?= 0
ifeq ($(MACHTYPE),aarch64)
	ARM := 1
	X64 := 0
endif

ARM_MACOS ?= 0
ifeq ($(shell uname), Darwin)
	ARM := 0
	ARM_MACOS := 1
endif

LINK_FORMAT := static
SHARED_LINK := 0

STATIC_LIB_X64 :=
STATIC_LIB_ARM :=
STATIC_LIB_ARM_CROSS :=
STATIC_LIB_ARM_MACOS :=

HOMEBREW_DIR := /opt/homebrew

CPP     := cpp -P
CC      := gcc
CFLAGS  := -g3 -O3 -Wall -msse3 -Wno-unknown-pragmas -fopenmp -funroll-loops -fcommon -I$(INCLUDE) -DCBLAS_GEMM $(EMAX_DEFINE)
ifeq ($(SAME_DISTANCE), 1)
CFLAGS  := -g3 -O3 -Wall -msse3 -Wno-unknown-pragmas -fopenmp -funroll-loops -fcommon -I$(INCLUDE) -DCBLAS_GEMM -DSAME_DISTANCE $(EMAX_DEFINE)
endif
LDFLAGS := -L/usr/lib64 -L/usr/local/lib -lm

ifeq ($(ARM),1)
CFLAGS  := -O1 -Wall -Wno-unknown-pragmas -funroll-loops -fopenmp -fcommon -I$(INCLUDE) -DARMZYNQ $(EMAX_DEFINE)
ifeq ($(SAME_DISTANCE), 1)
CFLAGS  := -O1 -Wall -Wno-unknown-pragmas -funroll-loops -fopenmp -fcommon -I$(INCLUDE) -DARMZYNQ -DSAME_DISTANCE $(EMAX_DEFINE)
endif
LDFLAGS := -L/usr/lib64 -L/usr/local/lib -lm -lrt -lX11 -lXext
CFLAGS_EMAX  := -O1 -Wall -Wno-unknown-pragmas -funroll-loops -fopenmp -fcommon -I$(INCLUDE) -I$(CONV) -DARMZYNQ $(EMAX_DEFINE)
CFLAGS_EMAX_NC  := -O1 -Wall -Wno-unknown-pragmas -funroll-loops -fopenmp -fcommon -I$(INCLUDE) -I$(CONV) -DARMZYNQ $(EMAX_DEFINE)
CFLAGS_EMAX_DMA  := -O1 -Wall -Wno-unknown-pragmas -funroll-loops -fopenmp -fcommon -I$(INCLUDE) -I$(CONV) -DARMZYNQ -DFPDDMA $(EMAX_DEFINE)
SRCS_EMAX := $(filter-out $(SRC_DIR)/sparse_imax.c, $(SRCS)) $(SRC_DIR)/sparse_imax-emax$(EMAX_VER).c
OBJS_EMAX := $(SRCS_EMAX:.c=.o)
endif

ifeq ($(ARM_MACOS),1)
CFLAGS := -g3 -O3 -Wall -Wno-unknown-pragmas -I$(HOMEBREW_DIR)/opt/libomp/include -Xpreprocessor -fopenmp -I$(INCLUDE) -DCBLAS_GEMM $(EMAX_DEFINE)
LDFLAGS := -L/usr/lib -L/usr/local/lib -L$(HOMEBREW_DIR)/opt/libomp/lib -lm -lomp
endif

ifeq ($(ARM_CROSS),1)
LDFLAGS := -L/usr/lib64 -L/usr/local/lib -lm -fopenmp -fcommon
endif
DEVICE_DEBUG := 0

ifeq ($(CPUONLY),1)
CFLAGS := -g3 -O3 -Wall -Wno-unknown-pragmas -fopenmp -funroll-loops -fcommon -I$(INCLUDE) -DUSE_MP
ifeq ($(SAME_DISTANCE), 1)
CFLAGS := -g3 -O3 -Wall -Wno-unknown-pragmas -fopenmp -funroll-loops -fcommon -I$(INCLUDE) -DUSE_MP -DSAME_DISTANCE
endif
endif

ifeq ($(CUDA),1)
CC   := nvcc
SRCS := $(wildcard $(SRC_DIR)/*.c)
SRCS_CU := $(wildcard $(SRC_DIR)/*.cu)
OBJS := $(SRCS:.c=.o) $(SRCS_CU:.cu=.o)
CFLAGS := -O3 -I$(INCLUDE) -Xcompiler -fcommon -DUSE_CUDA
ifeq ($(SAME_DISTANCE), 1)
CFLAGS := -O3 -I$(INCLUDE) -Xcompiler -fcommon -DUSE_CUDA -DSAME_DISTANCE
endif
LDFLAGS := -L/usr/lib64 -L/usr/local/lib -lm -lrt -lcusparse -lcublas
endif

all: $(PROGRAM)

$(PROGRAM): $(OBJS) $(MAIN_OBJS)
	$(CC) $(OBJS) $(MAIN_OBJS) -o $(PROGRAM) $(LDFLAGS) $(CFLAGS)

ifeq ($(ARM), 1)
$(PROGRAM).emax$(EMAX_VER): $(OBJS_EMAX) $(MAIN_OBJS)
	$(CC) $(OBJS_EMAX) $(MAIN_OBJS) -o $(PROGRAM).emax$(EMAX_VER) $(LDFLAGS) $(CFLAGS_EMAX)

$(PROGRAM).emax$(EMAX_VER)+dma: $(OBJS_EMAX) $(MAIN_OBJS)
	$(CC) $(OBJS_EMAX) $(MAIN_OBJS) -o $(PROGRAM).emax$(EMAX_VER)+dma $(LDFLAGS) $(CFLAGS_EMAX_DMA)

$(PROGRAM).emax$(EMAX_VER)+nc: $(OBJS) $(MAIN_OBJS)
	$(CC) $(OBJS) $(MAIN_OBJS) -o $(PROGRAM).emax$(EMAX_VER)+nc $(LDFLAGS) $(CFLAGS_EMAX_NC)
endif

$(TEST_SPARSE_PROGRAM): $(OBJS) $(TEST_SPARSE_OBJS)
	$(CC) $(OBJS) $(TEST_SPARSE_OBJS) -o $(TEST_SPARSE_PROGRAM) $(LDFLAGS) $(CFLAGS)

ifeq ($(ARM), 1)
$(TEST_SPARSE_PROGRAM).emax$(EMAX_VER): $(OBJS_EMAX) $(TEST_SPARSE_OBJS)
	$(CC) $(OBJS_EMAX) $(TEST_SPARSE_OBJS) -o $(TEST_SPARSE_PROGRAM).emax$(EMAX_VER) $(LDFLAGS) $(CFLAGS_EMAX)

$(TEST_SPARSE_PROGRAM).emax$(EMAX_VER)+dma: $(OBJS_EMAX) $(TEST_SPARSE_OBJS)
	$(CC) $(OBJS_EMAX) $(TEST_SPARSE_OBJS) -o $(TEST_SPARSE_PROGRAM).emax$(EMAX_VER)+dma $(LDFLAGS) $(CFLAGS_EMAX_DMA)

$(TEST_SPARSE_PROGRAM).emax$(EMAX_VER)+nc: $(OBJS) $(TEST_SPARSE_OBJS)
	$(CC) $(OBJS) $(TEST_SPARSE_OBJS) -o $(TEST_SPARSE_PROGRAM).emax$(EMAX_VER)+nc $(LDFLAGS) $(CFLAGS_EMAX_NC)
endif

$(TEST_DENSE_PROGRAM): $(OBJS) $(TEST_DENSE_OBJS)
	$(CC) $(OBJS) $(TEST_DENSE_OBJS) -o $(TEST_DENSE_PROGRAM) $(LDFLAGS) $(CFLAGS)

ifeq ($(ARM), 1)
$(TEST_DENSE_PROGRAM).emax$(EMAX_VER): $(OBJS_EMAX) $(TEST_DENSE_OBJS)
	$(CC) $(OBJS_EMAX) $(TEST_DENSE_OBJS) -o $(TEST_DENSE_PROGRAM).emax$(EMAX_VER) $(LDFLAGS) $(CFLAGS_EMAX)

$(TEST_DENSE_PROGRAM).emax$(EMAX_VER)+dma: $(OBJS_EMAX) $(TEST_DENSE_OBJS)
	$(CC) $(OBJS_EMAX) $(TEST_DENSE_OBJS) -o $(TEST_DENSE_PROGRAM).emax$(EMAX_VER)+dma $(LDFLAGS) $(CFLAGS_EMAX_DMA)

$(TEST_DENSE_PROGRAM).emax$(EMAX_VER)+nc: $(OBJS) $(TEST_DENSE_OBJS)
	$(CC) $(OBJS) $(TEST_DENSE_OBJS) -o $(TEST_DENSE_PROGRAM).emax$(EMAX_VER)+nc $(LDFLAGS) $(CFLAGS_EMAX_NC)
endif

CONV_EXE := ./conv-c2c/conv-c2c
ifeq ($(EMAX_VER), 7)
CONV_EXE := ./conv-c2d/conv-c2d
endif
ifeq ($(HARD_UNIT32), 1)
CONV_EXE := $(CONV_EXE) -u32
endif
$(SRC_DIR)/sparse_imax-emax$(EMAX_VER).c: $(SRC_DIR)/sparse_imax.c
	./conv-mark/conv-mark $< > $<-mark.c
	$(CPP) $(CFLAGS_EMAX_DMA) $<-mark.c > $<-cppo.c
	$(CONV_EXE) $<-cppo.c

.SUFFIXES: .o .c .cu

.c.o: $(HEADERS)
	@[ -d $(SRC_DIR) ]
	@[ -d $(TEST_DIR) ]
	$(CC) $(CFLAGS) -o $@ -c $<

.cu.o: $(HEADERS)
	@[ -d $(SRC_DIR) ]
	@[ -d $(TEST_DIR) ]
	$(CC) $(CFLAGS) -o $@ -c $<

clean:
	$(RM) *.o *.a *.so *.gch $(SRC_DIR)/*-*.c $(SRC_DIR)/*.o $(TEST_DIR)/*.o
