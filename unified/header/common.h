#ifndef __COMMON__
#define __COMMON__

#include <stdlib.h>
#include <stdio.h>
#include <string.h> //per il memcpy
#include <time.h>
#include <stdint.h>

/*
GUI: (https://wiki.libsdl.org/) (http://lazyfoo.net/tutorials/SDL)
- usa un solo rederer (non lo distrugge ogni volta)
- usa un solo rettangolo, lo riposiziona per colorare i pizel

Algo:
- dimensione righe multipla di 2 per poter usare and al posto del modulo

gcc omp.c -o omp -O2 -D_REENTRANT -lSDL2 -fopenmp && time ./omp
*/

typedef struct {
    uint8_t *matrix;
    uint32_t rows;
    uint32_t cols;
} GenState, *GenState_p;

void print_gen(GenState_p gen);
GenState_p create_gen(uint32_t rows, uint32_t cols);
void free_gen(GenState_p gen);
void clear_gen(GenState_p gen);
void random_gen(GenState_p gen);
void swap(void **a, void **b);
uint8_t countAliveCells(uint8_t *matrix, uint32_t x0, uint32_t x1, uint32_t x2, uint32_t y0, uint32_t y1, uint32_t y2);
uint8_t isPow2(uint32_t x);
uint32_t log2pow2(uint32_t x); //peggiora le performance
#endif