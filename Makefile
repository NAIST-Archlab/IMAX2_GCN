## EMAX5/6 Application Simulator       ##
##   Copyright (C) 2023 by NAIST UNIV. ##
##              Primary writer: D.Kim  ##
##          kim.dohyun.kg7@is.naist.jp ##
PROGRAM := imax_gcn
SRC_DIR := src
OBJS := main.o sparse.o layer.o utils.o
INCLUDE := ./include/
ifeq ($(MACHTYPE),x86_64)
	X64 ?= 1
	ARCH ?= X64
endif
ARM_CROSS ?= 0
ARM ?= 0
ifeq ($(MACHTYPE),aarch64)
	ARM := 1
	X64 := 0
endif

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
CFLAGS  := -g3 -O3 -Wall -msse3 -Wno-unknown-pragmas -fcommon -I$(INCLUDE) -DCBLAS_GEMM -DEMAX6 -DDEBUG
LDFLAGS := -L/usr/lib64 -L/usr/local/lib -lm -fopenmp -fcommon
#LDFLAGS := -L/usr/lib64 -L/usr/local/lib -L$(STATIC_LIB_X64) -lm

ifeq ($(ARM),1)
LDFLAGS := -L/usr/lib64 -L/usr/local/lib -lm -fopenmp -fcommon
# -L$(STATIC_LIB_ARM)
endif

ifeq ($(ARM_MACOS),1)
CFLAGS := -g3 -O3 -Wall -Wno-unknown-pragmas -I$(HOMEBREW_DIR)/opt/libomp/include -Xpreprocessor -fopenmp -I$(INCLUDE) -DCBLAS_GEMM -DEMAX6 -DDEBUG
LDFLAGS := -L/usr/lib -L/usr/local/lib -L$(HOMEBREW_DIR)/opt/libomp/lib -lm -lomp
# -L$(STATIC_LIB_ARM_MACOS)
endif

ifeq ($(ARM_CROSS),1)
LDFLAGS := -L/usr/lib64 -L/usr/local/lib -lm -fopenmp -fcommon
# -L$(STATIC_LIB_ARM_CROSS)
endif
DEVICE_DEBUG := 0

.SUFFIXES   := .c .o
$(PROGRAM): $(OBJS)
	$(CC) -o $(PROGRAM) $(LDFLAGS) $^

test_sparse: test_sparse.o sparse.o layer.o
	$(CC) -o test_sparse $(LDFLAGS) $^

%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	$(RM) *.o *.a *.so *.gch