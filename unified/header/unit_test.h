#include <sequential.h>
#include <openmp.h>
#include <cudax.h>
#include <assert.h>

void check_blinker(void (*compute_generation)(GenState_p s1, GenState_p s2));
void check_beehive(void (*compute_generation)(GenState_p s1, GenState_p s2));
void check_glinder(void (*compute_generation)(GenState_p s1, GenState_p s2));