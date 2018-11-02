#ifndef __GOL_CUDA__
#define __GOL_CUDA__

#include <common.h> 

typedef struct {
    uint8_t *matrix;
    uint32_t rows;
    uint32_t cols;
} GenStateGpu, *GenStateGpu_p;

void compute_cpu_generations_on_gpu(GenState_p s1, GenState_p s2, uint32_t iterations);

void compute_cpu_generations_on_gpu_multidim(GenState_p s1, GenState_p s2, uint32_t iterations);
void compute_cpu_generations_on_gpu_shared(GenState_p s1, GenState_p s2, uint32_t iterations);


#endif