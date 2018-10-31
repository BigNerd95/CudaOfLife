#include <common.h> 

void print_gen(GenState_p gen){
    printf("r: %d c: %d\n", gen->rows, gen->cols);

    for (uint32_t r = 0; r < gen->rows; r++){
        for (uint32_t c=0; c < gen->cols; c++){
            printf("%d ", gen->matrix[r * gen->cols + c]);
        }
        printf("\n");
    }
}

GenState_p create_gen(uint32_t rows, uint32_t cols){
    GenState_p gen = (GenState_p) malloc(sizeof(GenState));
    gen->matrix = (uint8_t *) malloc(cols * rows * sizeof(uint8_t));
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

GenState_p clone_gen(GenState_p src){
    GenState_p dst = create_gen(src->rows, src->cols);
    memcpy(dst->matrix, dst->matrix, src->cols * src->rows);
    return dst; 
} 

void random_gen(GenState_p gen){
    for (uint32_t i=0; i < gen->cols * gen->rows; i++){
        gen->matrix[i] = rand() % 2;
    }
}

void swap(void **a, void **b) {
    void *temp = *a;
    *a = *b;
    *b = temp;
}

uint8_t compare_gen(GenState_p gen1, GenState_p gen2){
    uint8_t result = 0;
    if(gen1->rows == gen2->rows && gen1->cols == gen2->cols){
        result = !memcmp(gen1->matrix, gen2->matrix, gen1->rows * gen1->cols);
    }
    return result;
}


uint8_t countAliveCells(uint8_t *matrix, uint32_t x0, uint32_t x1, uint32_t x2, uint32_t y0, uint32_t y1, uint32_t y2) {
    return matrix[x0 + y0] + matrix[x1 + y0] + matrix[x2 + y0] + matrix[x0 + y1]
         + matrix[x2 + y1] + matrix[x0 + y2] + matrix[x1 + y2] + matrix[x2 + y2];
}


uint8_t isPow2(uint32_t x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}


uint32_t log2pow2(uint32_t x){
    uint32_t pow2 = 0;
    while (x >>= 1){
        pow2++;
    }
    return pow2;
}