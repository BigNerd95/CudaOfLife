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

void get_execution_time_bidimensional(){
    printf("(Benchmark Gpu unidimensional) -> %lf\n", get_execution_time(MULTIDIM_R, MULTIDIM_C, 1000, (&compute_cpu_generations_on_gpu)));
    printf("(Benchmark Gpu bidimensional) -> %lf\n", get_execution_time(MULTIDIM_R, MULTIDIM_C, 1000, (&compute_cpu_generations_on_gpu_multidim)));    
}


void get_execution_time_shared(){
    uint32_t world_size= 4096;
    printf("\n(Benchmark Gpu con global con worls size: 4096) -> %lf\n", get_execution_time(world_size, world_size, 1000, (&compute_cpu_generations_on_gpu)));
    printf("\n(Benchmark Gpu con shared con worls size: 4096) -> %lf\n", get_execution_time(world_size, world_size, 1000, (&compute_cpu_generations_on_gpu_shared)));       
}


int benchmark_main(int argc, char *argv[]) {
    srand((unsigned) time(0));
    uint32_t iterations = 1000;
    get_execution_time_bidimensional();
    get_execution_time_shared();
     
    for (uint32_t world_size = 32; world_size <= 1000000000; world_size = world_size*2){        
            printf("\nWorld Size: %d \n", world_size);           
            printf("(Benchmark Cpu) -> %lf\n", get_execution_time(world_size, world_size, iterations, (&seq_compute_generations)));
            printf("(Benchmark OpenMp) -> %lf\n", get_execution_time(world_size, world_size, iterations, (&omp_compute_generations)));
            printf("(Benchmark Gpu) -> %lf\n", get_execution_time(world_size, world_size, iterations, (&compute_cpu_generations_on_gpu)));
    }
    return 0;
}