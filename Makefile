## EMAX5/6 Application Simulator       ##
##   Copyright (C) 2023 by NAIST UNIV. ##
##              Primary writer: D.Kim  ##
##          kim.dohyun.kg7@is.naist.jp ##
PROGRAM := imax_gcn
TEST_SPARSE_PROGRAM := test_sparse
SRC_DIR := src
TEST_DIR := test
INCLUDE := ./include/
SRCS := $(wildcard $(SRC_DIR)/*.c)
TEST_SRCS := $(wildcard $(TEST_DIR)/*.c)
OBJS := $(SRCS:.c=.o)
MAIN := main.c
MAIN_OBJS := main.o
NCHIP := 1
CPUONLY := 0
CUDA := 0
SAME_DISTANCE := 0
TEST_OBJS := $(TEST_SRCS:.c=.o)
HEADERS := $(INCLUDE)/emax6.h $(INCLUDE)/layer.h $(INCLUDE)/options.h $(INCLUDE)/sparse.h $(INCLUDE)/utils.h

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
CFLAGS  := -g3 -O3 -Wall -msse3 -Wno-unknown-pragmas -fopenmp -funroll-loops -fcommon -I$(INCLUDE) -DCBLAS_GEMM -DEMAX6 -DDEBUG -DUSE_IMAX2 -DUSE_MP -DNCHIP=$(NCHIP)
ifeq ($(SAME_DISTANCE), 1)
CFLAGS  := -g3 -O3 -Wall -msse3 -Wno-unknown-pragmas -fopenmp -funroll-loops -fcommon -I$(INCLUDE) -DCBLAS_GEMM -DEMAX6 -DDEBUG -DUSE_IMAX2 -DUSE_MP -DSAME_DISTANCE -DNCHIP=$(NCHIP)
endif
LDFLAGS := -L/usr/lib64 -L/usr/local/lib -lm

ifeq ($(ARM),1)
CFLAGS  := -O1 -Wall -Wno-unknown-pragmas -funroll-loops -fopenmp -fcommon -I$(INCLUDE) -DARMZYNQ -DEMAX6 -DDEBUG -DUSE_IMAX2 -DUSE_MP -DNCHIP=$(NCHIP)
ifeq ($(SAME_DISTANCE), 1)
CFLAGS  := -O1 -Wall -Wno-unknown-pragmas -funroll-loops -fopenmp -fcommon -I$(INCLUDE) -DARMZYNQ -DEMAX6 -DDEBUG -DUSE_IMAX2 -DUSE_MP -DSAME_DISTANCE -DNCHIP=$(NCHIP)
endif
LDFLAGS := -L/usr/lib64 -L/usr/local/lib -lm -lrt -lX11 -lXext
CFLAGS_EMAX6  := -O1 -Wall -Wno-unknown-pragmas -funroll-loops -fopenmp -fcommon -I$(INCLUDE) -DARMZYNQ -DEMAX6 -DUSE_IMAX2 -DUSE_MP -DNCHIP=$(NCHIP)
CFLAGS_EMAX6_DMA  := -O1 -Wall -Wno-unknown-pragmas -funroll-loops -fopenmp -fcommon -I$(INCLUDE) -DARMZYNQ -DEMAX6 -DFPDDMA -DUSE_IMAX2 -DUSE_MP -DNCHIP=$(NCHIP)
SRCS_EMAX6 := $(filter-out $(SRC_DIR)/sparse_imax.c, $(SRCS)) $(SRC_DIR)/sparse_imax-emax6.c
OBJS_EMAX6 := $(SRCS_EMAX6:.c=.o)
endif

ifeq ($(ARM_MACOS),1)
CFLAGS := -g3 -O3 -Wall -Wno-unknown-pragmas -I$(HOMEBREW_DIR)/opt/libomp/include -Xpreprocessor -fopenmp -I$(INCLUDE) -DCBLAS_GEMM -DEMAX6 -DDEBUG -DUSE_IMAX2 -DUSE_MP -DNCHIP=$(NCHIP)
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
CFLAGS := -O3 -I$(INCLUDE) -DUSE_CUDA
ifeq ($(SAME_DISTANCE), 1)
CFLAGS := -O3 -I$(INCLUDE) -DUSE_CUDA -DSAME_DISTANCE
endif
LDFLAGS := -L/usr/lib64 -L/usr/local/lib -lm -lrt -lcusparse -lcublas
endif

all: $(PROGRAM)

$(PROGRAM): $(OBJS) $(MAIN_OBJS)
	$(CC) $(OBJS) $(MAIN_OBJS) -o $(PROGRAM) $(LDFLAGS) $(CFLAGS)

ifeq ($(ARM), 1)
$(PROGRAM).emax6: $(OBJS_EMAX6) $(MAIN_OBJS)
	$(CC) $(OBJS_EMAX6) $(MAIN_OBJS) -o $(PROGRAM).emax6 $(LDFLAGS) $(CFLAGS_EMAX6)

$(PROGRAM).emax6+dma: $(OBJS_EMAX6) $(MAIN_OBJS)
	$(CC) $(OBJS_EMAX6) $(MAIN_OBJS) -o $(PROGRAM).emax6+dma $(LDFLAGS) $(CFLAGS_EMAX6_DMA)
endif

$(TEST_SPARSE_PROGRAM): $(OBJS) $(TEST_OBJS)
	$(CC) $(OBJS) $(TEST_OBJS) -o $(TEST_SPARSE_PROGRAM) $(LDFLAGS) $(CFLAGS)

ifeq ($(ARM), 1)
$(TEST_SPARSE_PROGRAM).emax6: $(OBJS_EMAX6) $(TEST_OBJS)
	$(CC) $(OBJS_EMAX6) $(TEST_OBJS) -o $(TEST_SPARSE_PROGRAM).emax6 $(LDFLAGS) $(CFLAGS_EMAX6)

$(TEST_SPARSE_PROGRAM).emax6+dma: $(OBJS_EMAX6) $(TEST_OBJS)
	$(CC) $(OBJS_EMAX6) $(TEST_OBJS) -o $(TEST_SPARSE_PROGRAM).emax6+dma $(LDFLAGS) $(CFLAGS_EMAX6_DMA)
endif

$(SRC_DIR)/sparse_imax-emax6.c: $(SRC_DIR)/sparse_imax.c
	./conv-mark/conv-mark $< > $<-mark.c
	$(CPP) $(CFLAGS_EMAX6_DMA) $<-mark.c > $<-cppo.c
	./conv-c2c/conv-c2c $<-cppo.c

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
