#ifndef __IMAX_H__
#define __IMAX_H__
#include <stdio.h>
#include <stdlib.h>

#if defined(EMAX7)
#include "../conv-c2d/emax7.h"
#elif defined(EMAX6)
#include "../conv-c2c/emax6.h"
#else
#ifndef UTYPEDEF
#define UTYPEDEF
typedef unsigned char Uchar;
typedef unsigned short Ushort;
typedef unsigned int Uint;
typedef unsigned long long Ull;
typedef long long int Sll;
#if __AARCH64EL__ == 1
typedef long double Dll;
#else
typedef struct { Ull u[2]; } Dll;
#endif
#endif
#endif

#endif