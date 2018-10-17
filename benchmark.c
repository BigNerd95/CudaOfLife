#define _POSIX_C_SOURCE 199309L

#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include <sys/time.h>   
#include <sys/resource.h> 
#include <string.h>


/*
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


void print_gen(GenState gen){
    printf("r: %d c: %d\n", gen.rows, gen.cols);

    for (int r = 0; r < gen.rows; r++){
        for (int c=0; c < gen.cols; c++){
            printf("%d ", gen.matrix[r * gen.cols + c]);
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

double get_execution_time(int rows, int cols, int iterations){
    struct timespec start, finish; 
    GenState_p s1 = create_gen(rows, cols);
    GenState_p s2 = create_gen(rows, cols);
    random_gen(s1);

    clock_gettime(CLOCK_REALTIME, &start); 
    
    for (int i = 0; i < iterations; i++) {
        compute_generation(s1, s2);
        swap((void *) &s1, (void *) &s2);
    }
    clock_gettime(CLOCK_REALTIME, &finish);
    double seconds = (double)(finish.tv_sec - start.tv_sec); 
    double ns = (double)(finish.tv_nsec - start.tv_nsec); 
         
    if (start.tv_nsec > finish.tv_nsec) { // clock underflow 
	    --seconds; 
	    ns += 1000000000; 
    } 
    double total_time = seconds + ns/1000000000.0; 
    free_gen(s1);
    free_gen(s2);
    return total_time;
}

int get_iterations_per_second(int rows, int cols, int timer_seconds){
    int msec = 0, trigger = timer_seconds * 1000; /* trigger is in ms */
    clock_t before = clock();
    GenState_p s1 = create_gen(rows, cols);
    GenState_p s2 = create_gen(rows, cols);
    random_gen(s1);
    long int iterations = 0;

    do {
        compute_generation(s1, s2);
        swap((void *) &s1, (void *) &s2);

        clock_t difference = clock() - before;
        msec = difference * 1000 / CLOCKS_PER_SEC;
        iterations++;
    } while ( msec < trigger );

    printf("For size %d Time taken %d seconds %d milliseconds (%ld iterations)\n", rows, msec/1000, msec%1000, iterations);
    free_gen(s1);
    free_gen(s2);
}

int main(int argc, char *argv[]) {
    srand((unsigned) time(0));
    int word_size = 32;
    for (int i = 1; i < 8; i++){        
        printf("\nWord size: %d. Amount of time: %lf\n", word_size, get_execution_time(word_size, word_size, 10000));
        //get_iterations_per_second(word_size, word_size, 1);
        word_size  *= 2;
    }
    return 0;
}
