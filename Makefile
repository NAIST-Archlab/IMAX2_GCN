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
CFLAGS  := -g3 -O3 -Wall -msse3 -Wno-unknown-pragmas -fopenmp -funroll-loops -fcommon -I$(INCLUDE) -DCBLAS_GEMM -DEMAX6 -DDEBUG
LDFLAGS := -L/usr/lib64 -L/usr/local/lib -lm

ifeq ($(ARM),1)
CFLAGS  := -O1 -Wall -Wno-unknown-pragmas -funroll-loops -fopenmp -fcommon -I$(INCLUDE) -DARMZYNQ -DEMAX6
LDFLAGS := -L/usr/lib64 -L/usr/local/lib -lm -lrt -lX11 -lXext
PROGRAM := imax_gcn.emax6
TEST_SPARSE_PROGRAM := test_sparse.emax6
SRCS := $(filter-out $(SRC_DIR)/sparse_imax.c, $(SRCS)) $(SRC_DIR)/sparse_imax-emax6.c
TEST_SRCS := $(wildcard $(TEST_DIR)/*.c) $(SRC_DIR)/sparse_imax-emax6.c
#TEST_SRCS := $(wildcard $(TEST_DIR)/*.c)
OBJS := $(SRCS:.c=.o)
endif

ifeq ($(ARM_MACOS),1)
CFLAGS := -g3 -O3 -Wall -Wno-unknown-pragmas -I$(HOMEBREW_DIR)/opt/libomp/include -Xpreprocessor -fopenmp -I$(INCLUDE) -DCBLAS_GEMM -DEMAX6 -DDEBUG
LDFLAGS := -L/usr/lib -L/usr/local/lib -L$(HOMEBREW_DIR)/opt/libomp/lib -lm -lomp
endif

ifeq ($(ARM_CROSS),1)
LDFLAGS := -L/usr/lib64 -L/usr/local/lib -lm -fopenmp -fcommon
endif
DEVICE_DEBUG := 0

all: $(PROGRAM)

$(PROGRAM): $(OBJS) $(MAIN_OBJS)
	$(CC) $(OBJS) $(MAIN_OBJS) -o $(PROGRAM) $(LDFLAGS) $(CFLAGS)

$(TEST_SPARSE_PROGRAM): $(OBJS) $(TEST_OBJS)
	$(CC) $(OBJS) $(TEST_OBJS) -o $(TEST_SPARSE_PROGRAM) $(LDFLAGS) $(CFLAGS)

$(SRC_DIR)/sparse_imax-emax6.c: $(SRC_DIR)/sparse_imax.c
	./conv-mark/conv-mark $< > $<-mark.c
	$(CPP) $(CFLAGS) $<-mark.c > $<-cppo.c
	./conv-c2c/conv-c2c $<-cppo.c

.c.o: $(HEADERS)
	@[ -d $(SRC_DIR) ]
	@[ -d $(TEST_DIR) ]
	$(CC) $(CFLAGS) -o $@ -c $<

clean:
	$(RM) *.o *.a *.so *.gch $(SRC_DIR)/*-*.c $(SRC_DIR)/*.o $(TEST_DIR)/*.o