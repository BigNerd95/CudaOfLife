#ifndef __COMMON__
#define __COMMON__

#include <stdlib.h>
#include <stdio.h>
#include <string.h> //per il memcpy
#include <time.h> 

/*
GUI: (https://wiki.libsdl.org/) (http://lazyfoo.net/tutorials/SDL)
- usa un solo rederer (non lo distrugge ogni volta)
- usa un solo rettangolo, lo riposiziona per colorare i pizel

Algo:
- dimensione righe multipla di 2 per poter usare and al posto del modulo

gcc omp.c -o omp -O2 -D_REENTRANT -lSDL2 -fopenmp && time ./omp
*/

typedef struct {
    unsigned char *matrix;
    unsigned int rows;
    unsigned int cols;
} GenState, *GenState_p;

void print_gen(GenState_p gen);
GenState_p create_gen(int rows, int cols);
void free_gen(GenState_p gen);
void clear_gen(GenState_p gen);
void random_gen(GenState_p gen);
void swap(void **a, void **b);
unsigned char countAliveCells(unsigned char *matrix, size_t x0, size_t x1, size_t x2, size_t y0, size_t y1, size_t y2);
int isPow2(unsigned int x);
unsigned int log2pow2(unsigned int x); //peggiora le performance
#endif