#ifndef __GOL_OMP__
#define __GOL_OMP__

#include <common.h> 
#include <omp.h>

void omp_compute_generations(GenState_p s1, GenState_p s2, uint32_t iterations);

#endif