#include <common.h> 

/*
GUI: (https://wiki.libsdl.org/) (http://lazyfoo.net/tutorials/SDL)
- usa un solo rederer (non lo distrugge ogni volta)
- usa un solo rettangolo, lo riposiziona per colorare i pizel

Algo:
- dimensione righe multipla di 2 per poter usare and al posto del modulo

gcc omp.c -o omp -O2 -D_REENTRANT -lSDL2 -fopenmp && time ./omp
*/

void print_gen(GenState_p gen){
    printf("r: %d c: %d\n", gen->rows, gen->cols);

    for (int r = 0; r < gen->rows; r++){
        for (int c=0; c < gen->cols; c++){
            printf("%d ", gen->matrix[r * gen->cols + c]);
        }
        printf("\n");
    }
}

GenState_p create_gen(int rows, int cols){
    GenState_p gen = (GenState_p) malloc(sizeof(GenState));
    gen->matrix = (unsigned char *) malloc(cols * rows * sizeof(unsigned char));
    gen->rows = rows;
    gen->cols = cols;
    return gen;
}

void free_gen(GenState_p gen){
    free(gen->matrix);
    free(gen);
}

void clear_gen(GenState_p gen){
    memset(gen->matrix, 0, gen->cols * gen->rows);
}

void random_gen(GenState_p gen){
    for (int i=0; i < gen->cols * gen->rows; i++){
        gen->matrix[i] = rand() % 2;
    }
}

void swap(void **a, void **b) {
    void *temp = *a;
    *a = *b;
    *b = temp;
}

unsigned char countAliveCells(unsigned char *matrix, size_t x0, size_t x1, size_t x2, size_t y0, size_t y1, size_t y2) {
    return matrix[x0 + y0] + matrix[x1 + y0] + matrix[x2 + y0] + matrix[x0 + y1]
         + matrix[x2 + y1] + matrix[x0 + y2] + matrix[x1 + y2] + matrix[x2 + y2];
}


int isPow2(unsigned int x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}


//peggiora le performance
unsigned int log2pow2(unsigned int x){
    unsigned int pow2 = 0;
    while (x >>= 1){
        pow2++;
    }
    return pow2;
}
