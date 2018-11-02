#ifndef __GOL_COMMON__
#define __GOL_COMMON__

#include <stdlib.h>
#include <stdio.h>
#include <string.h> //per il memcpy
#include <time.h>
#include <stdint.h>


#define MULTIDIM_R 4096
#define MULTIDIM_C 4096

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
uint8_t compare_gen(GenState_p gen1, GenState_p gen2);
GenState_p clone_gen(GenState_p src);
uint8_t countAliveCells(uint8_t *matrix, uint32_t x0, uint32_t x1, uint32_t x2, uint32_t y0, uint32_t y1, uint32_t y2);
uint8_t isPow2(uint32_t x);
uint32_t log2pow2(uint32_t x);

#endif