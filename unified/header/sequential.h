#ifndef __GOL_SEQ__
#define __GOL_SEQ__
#include <common.h> 

void compute_generations_singlefor(GenState_p s1, GenState_p s2, uint32_t iterations);
void compute_generations(GenState_p s1, GenState_p s2, uint32_t iterations);
void compute_generations_pow2(GenState_p s1, GenState_p s2, uint32_t iterations);
#endif