#include <common.h> 
#include <omp.h>

void omp_compute_generations_singlefor(GenState_p s1, GenState_p s2, uint32_t iterations);
void omp_compute_generations(GenState_p s1, GenState_p s2, uint32_t iterations);
void omp_compute_generations_pow2(GenState_p s1, GenState_p s2, uint32_t iterations);
