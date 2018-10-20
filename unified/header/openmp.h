#ifndef __COMMON__
	#include <common.h> 
#endif
#include <omp.h>

void omp_compute_generation_singlefor(GenState_p s1, GenState_p s2);
void omp_compute_generation(GenState_p s1, GenState_p s2);
void omp_compute_generation_pow2(GenState_p s1, GenState_p s2);