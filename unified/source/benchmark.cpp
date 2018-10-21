#include <benchmark.h>

double get_execution_time(uint32_t rows, uint32_t cols, uint32_t iterations, void (*compute_generation)(GenState_p s1, GenState_p s2, uint32_t iterations)){
    struct timespec start, finish; 
    GenState_p s1 = create_gen(rows, cols);
    GenState_p s2 = create_gen(rows, cols);
    random_gen(s1);

    clock_gettime(CLOCK_REALTIME, &start); 
    
    (*compute_generation)(s1, s2, iterations);
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

/*
int main(int argc, char *argv[]) {
    srand((unsigned) time(0));
    uint32_t word_size = 1024;
    
    for (int i = 1; i < 8; i++){        
        printf("\n Benchmark Cpu seriale\n");
        printf("\nWord size: %d. Amount of time: %lf\n", word_size, get_execution_time(word_size, word_size, 10000, (&compute_generations_singlefor)));
        printf("\n Benchmark Cpu OpernMp\n");
        printf("\nWord size: %d. Amount of time: %lf\n", word_size, get_execution_time(word_size, word_size, 10000, (&omp_compute_generations_singlefor)));
        printf("\nBenchmark Gpu\n");
        printf("\nWord size: %d. Amount of time: %lf\n", word_size, get_execution_time(word_size, word_size, 10000, (&compute_cpu_generations_on_gpu)));
        word_size  *= 2;
    }
    return 0;
}*/