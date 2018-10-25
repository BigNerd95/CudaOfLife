#define _POSIX_C_SOURCE 199309L

#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include <sys/time.h>   
#include <sys/resource.h> 
#include <string.h>
#include <unistd.h>
#include <assert.h>

/*
gcc unit_test.c -o unit_test -I/usr/include/SDL2 -D_REENTRANT -lSDL2 | ./unit_test


GUI: (https://wiki.libsdl.org/) (http://lazyfoo.net/tutorials/SDL)
- usa un solo rederer (non lo distrugge ogni volta)
- usa un solo rettangolo, lo riposiziona per colorare i pizel

Algo:
- dimensione righe multipla di 2 per poter usare and al posto del modulo
*/

#define WIDTH 640
#define HEIGHT 640
#define DELAY 50

typedef struct {
    unsigned char *matrix;
    unsigned int rows;
    unsigned int cols;
} GenState, *GenState_p;


void print_gen(GenState_p gen){
    //printf("r: %d c: %d\n", gen->rows, gen->cols);

    for (int r = 0; r < gen->rows; r++){
        for (int c=0; c < gen->cols; c++){
            if(gen->matrix[r * gen->cols + c])
                printf("O");
            else 
                printf(".");
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

void compute_generation(GenState_p s1, GenState_p s2){
    for (size_t y = 0; y < s1->rows; y++) {
        size_t y0 = ((y + s1->rows - 1) % s1->rows) * s1->cols;
        size_t y1 = y * s1->cols;
        size_t y2 = ((y + 1) % s1->rows) * s1->cols;

        for (size_t x = 0; x < s1->cols; x++) {
            size_t x0 = (x + s1->cols - 1) % s1->cols;
            size_t x2 = (x + 1) % s1->cols;

            unsigned char aliveCells = countAliveCells(s1->matrix, x0, x, x2, y0, y1, y2);
            s2->matrix[y1 + x] = (aliveCells == 3 || (aliveCells == 2 && s1->matrix[x + y1])) ? 1 : 0;
        }
    }
}
void check_blinker(){
    GenState_p blinker = create_gen(8, 8);
    GenState_p blinker2 = create_gen(8, 8);
    clear_gen(blinker);

    blinker->matrix[19] = 1;
    blinker->matrix[27] = 1;
    blinker->matrix[35] = 1;
    printf("\nPrint blinker state 1\n ");
    print_gen(blinker);

    compute_generation(blinker, blinker2);
    printf("\nPrint blinker state 2 \n");
    print_gen(blinker2);

    for (int i = 0; i < 64; i++){
        if (i == 26 || i == 27 || i == 28) 
            assert(blinker2->matrix[i] == 1);
        else
            assert(blinker2->matrix[i] == 0);  
    }

    free_gen(blinker);
    free_gen(blinker2);
}

void check_beehive(){
    GenState_p beehive = create_gen(8, 8);
    GenState_p beehive2 = create_gen(8, 8);

    clear_gen(beehive);
    beehive->matrix[19] = 1;
    beehive->matrix[20] = 1;
    beehive->matrix[26] = 1;
    beehive->matrix[29] = 1;
    beehive->matrix[35] = 1;
    beehive->matrix[36] = 1;
    printf("\nPrint beehive state 1\n");
    print_gen(beehive);

    compute_generation(beehive, beehive2);
    printf("\nPrint beehive state 2\n");
    print_gen(beehive2);
    
    for (int i = 0; i < 64; i++){
        if (i == 19 || i == 20 || i == 26 || i == 29 || i == 35 || i == 36) 
            assert(beehive2->matrix[i] == 1);
        else
            assert(beehive2->matrix[i] == 0);  
    }

    free_gen(beehive);
    free_gen(beehive2);
}

void check_glinder(){
    GenState_p glinder = create_gen(8, 8);
    GenState_p glinder2 = create_gen(8, 8);

    clear_gen(glinder);
    glinder->matrix[11] = 1;
    glinder->matrix[17] = 1;
    glinder->matrix[19] = 1;
    glinder->matrix[26] = 1;
    glinder->matrix[27] = 1;
    printf("\nPrint glinder state 1\n");
    print_gen(glinder);

    compute_generation(glinder, glinder2);
    printf("\nPrint glinder state 2\n");
    print_gen(glinder2);  
    for (int i = 0; i < 64; i++){
        if (i == 10 || i == 19 || i == 20 || i == 26 || i == 27) 
            assert(glinder2->matrix[i] == 1);
        else
            assert(glinder2->matrix[i] == 0);  
    }

    compute_generation(glinder2, glinder);
    printf("\nPrint glinder state 3\n");
    print_gen(glinder);
    for (int i = 0; i < 64; i++){
        if (i == 11 || i == 20 || i == 26 || i == 27 || i == 28) 
            assert(glinder->matrix[i] == 1);
        else
            assert(glinder->matrix[i] == 0);  
    }

    compute_generation(glinder, glinder2);
    printf("\nPrint glinder state 4\n");
    print_gen(glinder2);  
    for (int i = 0; i < 64; i++){
        if (i == 18 || i == 20 || i == 27 || i == 28 || i == 35) 
            assert(glinder2->matrix[i] == 1);
        else
            assert(glinder2->matrix[i] == 0);  
    }

    compute_generation(glinder2, glinder);
    printf("\nPrint glinder state 1.2\n");
    print_gen(glinder);  
    for (int i = 0; i < 64; i++){
        if (i == 20 || i == 26 || i == 28 || i == 35 || i == 36) 
            assert(glinder->matrix[i] == 1);
        else
            assert(glinder->matrix[i] == 0); 
    }
    
    free_gen(glinder);
    free_gen(glinder2);
}

void execute(int rows, int cols, int iterations){
    GenState_p s1 = create_gen(rows, cols);
    GenState_p s2 = create_gen(rows, cols);
    random_gen(s1);
    
    for (int i = 0; i < iterations; i++) {
        compute_generation(s1, s2);
        swap((void *) &s1, (void *) &s2);
    }
    free_gen(s1);
    free_gen(s2);
}


int main(int argc, char *argv[]) {
    srand((unsigned) time(0));
    int word_size = 32;
    assert(word_size <= 32);
    check_beehive();
    check_blinker();
    check_glinder();
    puts("Eseguito correttamente");
    execute(word_size, word_size, 1000);
    return 0;
}
