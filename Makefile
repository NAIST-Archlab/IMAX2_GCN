## EMAX5/6 Application Simulator       ##
##   Copyright (C) 2023 by NAIST UNIV. ##
##              Primary writer: D.Kim  ##
##          kim.dohyun.kg7@is.naist.jp ##

SUFFIX   := .o.c

PROGRAM := imax_gcn
SRC_DIR := src
OBJS := $(SRC_DIR)/main.o $(SRC_DIR)/sparse_kernel/sparse.o $(SRC_DIR)/layer.o
INCLUDE := $(wildcard ./include/*.h) 
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
	ifeq ($(ARM), 1)
		ARM := 0
		ARM_MACOS := 1
	endif
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
CFLAGS  := -g3 -O0 -Wall -msse3 -Wno-unknown-pragmas -I$(INCLUDE) -DCBLAS_GEMM -DEMAX6 -DDEBUG
LDFLAGS := -L/usr/lib64 -L/usr/local/lib -L$(STATIC_LIB_X64) -lm

ifeq ($(ARM),1)
LDFLAGS := -L/usr/lib64 -L/usr/local/lib -L$(STATIC_LIB_ARM) -lm
endif

ifeq ($(ARM_MACOS),1)
CFLAGS := -g3 -O0 -Wall -msse3 -Wno-unknown-pragmas -I$(HOMEBREW_DIR)/include -I$(INCLUDE) -DCBLAS_GEMM -DEMAX6 -DDEBUG
LDFLAGS := -L/usr/lib -L/usr/local/lib -L$(HOMEBREW_DIR)/lib -L$(STATIC_LIB_ARM_MACOS) -lm 
endif

ifeq ($(ARM_CROSS),1)
LDFLAGS := -L/usr/lib64 -L/usr/local/lib -L$(STATIC_LIB_ARM_CROSS) -lm
endif
DEVICE_DEBUG := 0

$(PROGRAM): $(OBJS)
	$(CC) -o $(PROGRAM) $^

.o.c:
	$(CC) $(CFLAGS) $<

clean:
	$(RM) *.o *.a *.so