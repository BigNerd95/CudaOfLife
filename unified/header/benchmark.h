#define _POSIX_C_SOURCE 199309L

#include <openmp.h>
#include <sequential.h>  
#include <sys/time.h>   
#include <sys/resource.h> 

void get_iterations_per_second(uint32_t rows, uint32_t cols, uint32_t timer_seconds, void (*compute_generation)(GenState_p s1, GenState_p s2));
double get_execution_time(uint32_t rows, uint32_t cols, uint32_t iterations, void (*compute_generation)(GenState_p s1, GenState_p s2));
