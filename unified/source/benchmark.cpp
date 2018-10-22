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

int main(int argc, char *argv[]) {
    srand((unsigned) time(0));
    
    for (uint32_t world_size = 32; world_size <= 1024; world_size = world_size*2){        
        for (uint32_t iterations = 1000; iterations <= 1000000; iterations = iterations*10){        
            printf("\n(Benchmark Gpu) -> Word size: %d -> Number of iterations: %d, Amount of time: %lf\n", world_size, iterations, get_execution_time(world_size, world_size, 10000, (&compute_cpu_generations_on_gpu)));
            printf("\n(Benchmark OpenMp) -> Word size: %d -> Number of iterations: %d, Amount of time: %lf\n", world_size, iterations, get_execution_time(world_size, world_size, 10000, (&omp_compute_generations_singlefor)));
            printf("\n(Benchmark Sequential) -> Word size: %d -> Number of iterations: %d, Amount of time: %lf\n", world_size, iterations, get_execution_time(world_size, world_size, 10000, (&compute_generations_singlefor)));
        }
    }
    
    return 0;
}