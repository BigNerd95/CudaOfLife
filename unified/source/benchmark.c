#include <benchmark.h>

void get_iterations_per_second(uint32_t rows, uint32_t cols, uint32_t timer_seconds, void (*compute_generation)(GenState_p s1, GenState_p s2)){
    uint32_t msec = 0, trigger = timer_seconds * 1000; /* trigger is in ms */
    clock_t before = clock();
    GenState_p s1 = create_gen(rows, cols);
    GenState_p s2 = create_gen(rows, cols);
    random_gen(s1);
    uint32_t iterations = 0;

    do {
        (*compute_generation)(s1, s2);
        swap((void **) &s1, (void **) &s2);
        iterations++;
        clock_t difference = clock() - before; //quanti cicli di clock sono passati
        msec = difference * 1000 / CLOCKS_PER_SEC; //converto i cicli di clock in millisecondi
    } while ( msec < trigger );

    printf("For size %d Time taken %d seconds %d milliseconds (%d iterations)\n", rows, msec/1000, msec%1000, iterations);
    free_gen(s1);
    free_gen(s2);
}


/*
int main(int argc, char *argv[]) {
    srand((unsigned) time(0));
    uint32_t word_size = 128;
    get_iterations_per_second(word_size, word_size, 3, (&compute_generation_singlefor));
    get_iterations_per_second(word_size, word_size, 3, (&compute_generation));
    get_iterations_per_second(word_size, word_size, 3, (&compute_generation_pow2));

    //for (int i = 1; i < 8; i++){        
        //printf("\nWord size: %d. Amount of time: %lf\n", word_size, get_execution_time(word_size, word_size, 10000));
        //printf("\n Cpu seriale\n");
        //get_iterations_per_second(word_size, word_size, 3, (&compute_generation));
        //printf("\n OpenMp single for\n");
        //get_iterations_per_second(word_size, word_size, 3, (&compute_generation_singlefor));
        //printf("\n OpenMp double for\n");
        //get_iterations_per_second(word_size, word_size, 3, (&compute_generation_doublefor));
     //   word_size  *= 2;
   // }
    return 0;
}*/

/*double get_execution_time(uint32_t rows, uint32_t cols, uint32_t iterations, void (*compute_generation)(GenState_p s1, GenState_p s2)){
    struct timespec start, finish; 
    GenState_p s1 = create_gen(rows, cols);
    GenState_p s2 = create_gen(rows, cols);
    random_gen(s1);

    clock_gettime(CLOCK_REALTIME, &start); 
    
    for (int i = 0; i < iterations; i++) {
        (*compute_generation)(s1, s2);
        swap((void **) &s1, (void **) &s2);
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
*/