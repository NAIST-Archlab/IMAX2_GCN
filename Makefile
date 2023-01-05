## EMAX5/6 Application Simulator       ##
##   Copyright (C) 2021 by NAIST UNIV. ##
##         Primary writer: T.Sugahara##
##                sugahara.takuya.ss4@is.naist.jp ##

ARCHIVE := $(AR) $(ARFLAGS)

SUFFIX   := .c

STATIC_LIB_CENT := $(wildcard ./lib/cent)
STATIC_LIB_ARM := $(wildcard ./lib/arm)
STATIC_LIB_ARM_CROSS := $(wildcard ./lib/arm_cross)

EMAX6_INCLUDE := $(wildcard ./include/*.h) 
ifeq ($(MACHTYPE),x86_64)
	CENT ?= 1
	ARCH ?= CENT
endif
ARM_CROSS ?= 0
ARM ?= 0
ifeq ($(MACHTYPE),aarch64)
	ARM := 1
	CENT := 0
endif
LINK_FORMAT := static
SHARED_LINK := 0

CPP     := cpp -P
CC      := gcc
CFLAGS  := -g3 -O0 -Wall  -msse3 -Wno-unknown-pragmas -funroll-loops -I/usr/local/include -I/usr/include/openblas -I$(EMAX6_INCLUDE) -DCBLAS_GEMM -DEMAX6 -DDEBUG
LDFLAGS := -L/usr/lib64 -L/usr/local/lib -L/usr/lib64/atlas -L$(STATIC_LIB_CENT) -lm -lrt -lopenblas -lX11 -lXext -lsparse
ifeq ($(ARM),1)
LDFLAGS := -L/usr/lib64 -L/usr/local/lib -L/usr/lib64/atlas -L$(STATIC_LIB_ARM) -lm -lrt -lopenblas -lX11 -lXext -lsparse
endif

ifeq ($(ARM_CROSS),1)
LDFLAGS := -L/usr/lib64 -L/usr/local/lib -L/usr/lib64/atlas -L$(STATIC_LIB_ARM_CROSS) -lm -lrt -lopenblas -lX11 -lXext -lsparse
endif
DEVICE_DEBUG := 0

ifeq ($(ARM),1)
	test: binary test1 $(EMAX6_INCLUDE)

test1: $(EMAX6_SPARSE_TEST)
	(cd ./test; $(MAKE) -f Makefile.arm LINK_FORMAT=$(LINK_FORMAT) CFLAGS="$(CFLAGS)")
endif

ifeq ($(CENT),1)
	test: binary test1  $(EMAX6_INCLUDE)

test1: $(EMAX6_SPARSE_TEST)
	(cd ./test; $(MAKE) -f Makefile.cent LINK_FORMAT=$(LINK_FORMAT) CFLAGS="$(CFLAGS)")
endif


######################## test #######################################################

#./../csim/csim -x test/test_chipB_div+dma
# xdisp系をcsimから消した
ifeq ($(ARM_CROSS),1)
	test: binary test1 $(EMAX6_INCLUDE)

test1: $(EMAX6_SPARSE_TEST)
	(cd ./test; $(MAKE) -f Makefile.csim LINK_FORMAT=$(LINK_FORMAT) CFLAGS="$(CFLAGS)")
endif

ifeq ($(GPU),1)
	test: 
	(cd ./test; $(MAKE) -f Makefile.gpu)
endif
########################################################################################


ifeq ($(LINK_FORMAT),shared)
	binary: shared 
else
	binary: static
endif


static: $(AR_TARGET)
shared: $(SHARED_TARGET)


############################## archive #################################################
ifeq ($(ARM),1)
	$(AR_TARGET):  $(EMAX6_SPARSE_ARM_LIB) $(EMAX6_INCLUDE)
	$(ARCHIVE)  $@ $(EMAX6_SPARSE_ARM_LIB)
	$(RM) ./kernel/*-*.c
endif
ifeq ($(CENT),1)
	$(AR_TARGET): $(EMAX6_SPARSE_LIB) $(EMAX6_INCLUDE)
	$(ARCHIVE)  $@ $(EMAX6_SPARSE_LIB)
endif
ifeq ($(ARM_CROSS),1)
	$(AR_TARGET):  $(EMAX6_SPARSE_ARM_CROSS_LIB) $(EMAX6_INCLUDE)
	$(ARCHIVE)  $@ $(EMAX6_SPARSE_ARM_CROSS_LIB)
	$(RM) ./kernel/*-*.c
endif
#######################################################################################


########################  3rd party  ##################################################
ifeq ($(ARM),1)
	./build_arm/3rd/Matrix_Format_Io/%.o: ./3rd/Matrix_Format_Io/%.c $(MAT_FORMAT_INCLUDE) 
	$(CC) $(CFLAGS) -c  $< -o $@
endif

ifeq ($(CENT),1)
	./build_cent/3rd/Matrix_Format_Io/%.o: ./3rd/Matrix_Format_Io/%.c $(MAT_FORMAT_INCLUDE) 
	$(CC) $(CFLAGS) -c  $< -o $@
endif

ifeq ($(ARM_CROSS),1)
	./build_arm_cross/3rd/Matrix_Format_Io/%.o: ./3rd/Matrix_Format_Io/%.c $(MAT_FORMAT_INCLUDE) 
	$(CC) $(CFLAGS) -c  $< -o $@
endif
########################################################################################

########################  util  ########################################################
ifeq ($(ARM),1)
	$(SHARED_TARGET): $(EMAX6_SPARSE_UTIL_ARM_SRC)  $(EMAX6_INCLUDE)
	$(CC) $(SHARED_FLAG) $(CFLAGS) -o $@ $^

./build_arm/util/%.o: ./util/%.c ./include/emax6_sparselib.h 
	$(CC) $(CFLAGS) -c  $< -o $@
endif

ifeq ($(CENT),1)
	$(SHARED_TARGET): $(EMAX6_SPARSE_UTIL_SRC) $(EMAX6_INCLUDE)
	$(CC) $(SHARED_FLAG) $(CFLAGS) -o $@ $^
	./build_cent/util/%.o: ./util/%.c ./include/emax6_sparselib.h 
	$(CC) $(CFLAGS) -c  $< -o $@
endif

ifeq ($(ARM_CROSS),1)
	$(SHARED_TARGET): $(EMAX6_SPARSE_UTIL_ARM_SRC)  $(EMAX6_INCLUDE)
	$(CC) $(SHARED_FLAG) $(CFLAGS) -o $@ $^
	./build_arm_cross/util/%.o: ./util/%.c ./include/emax6_sparselib.h 
	$(CC) $(CFLAGS) -c  $< -o $@
endif
########################################################################################



############################### kernel #################################################
ifeq ($(ARM),1)
	ifeq ($(LINK_FORMAT),shared)
	./build_arm/kernel/%-emax6.c: ./kernel/%.c ./include/emax6_sparselib.h
	perl ./conv-mark/conv-mark $< > $<-mark.c
	$(CPP) $(OPTION) $(INCLUDE)  $<-mark.c > $<-cppo.c
	./conv-c2c/conv-c2c.arm $<-cppo.c
else
	./build_arm/kernel/%-emax6.o: ./kernel/%.c  ./include/emax6_sparselib.h
	perl ./conv-mark/conv-mark $< > $<-mark.c
	$(CPP) $(OPTION) $(INCLUDE)  $<-mark.c > $<-cppo.c
	./conv-c2c/conv-c2c.arm $<-cppo.c
	$(CC) $(CFLAGS) -c  $(basename $<)-emax6.c -o $@
endif
endif


ifeq ($(CENT),1)
	./build_cent/kernel/%.o: ./kernel/%.c ./include/emax6_sparselib.h 
	$(CC) $(CFLAGS) -c  $< -o $@
endif


ifeq ($(ARM_CROSS),1)
	ifeq ($(LINK_FORMAT),shared)
	./build_arm_cross/kernel/%-emax6.c: ./kernel/%.c ./include/emax6_sparselib.h
	perl ./conv-mark/conv-mark $< > $<-mark.c
	$(CPP) $(OPTION) $(INCLUDE)  $<-mark.c > $<-cppo.c
	./conv-c2c/conv-c2c.csim $<-cppo.c
else
	./build_arm_cross/kernel/%-emax6.o: ./kernel/%.c  ./include/emax6_sparselib.h
	perl ./conv-mark/conv-mark $< > $<-mark.c
	$(CPP) $(OPTION) $(INCLUDE)  $<-mark.c > $<-cppo.c
	./conv-c2c/conv-c2c.csim $<-cppo.c
	$(CC) $(CFLAGS) -c  $(basename $<)-emax6.c -o $@
endif
endif
########################################################################################



make_build_dir:
	@mkdir -p ${CURDIR}/build_arm/kernel
	@mkdir -p ${CURDIR}/build_arm/util
	@mkdir -p ${CURDIR}/build_cent/kernel
	@mkdir -p ${CURDIR}/build_cent/util
	@mkdir -p ${CURDIR}/build_arm_cross/kernel
	@mkdir -p ${CURDIR}/build_arm_cross/util



ifeq ($(CENT),1)
	clean:
	$(RM) *.o *.a *.so
endif


ifeq ($(ARM),1)
	clean:
	$(RM) *.o *.a *.so
endif


ifeq ($(ARM_CROSS),1)
	clean:
	$(RM) *.o *.a *.so
endif
