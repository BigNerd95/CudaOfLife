#include <common.h> 

typedef struct {
    unsigned char *matrix;
    unsigned int rows;
    unsigned int cols;
} GenStateGpu, *GenStateGpu_p;

/*
__device__ unsigned char countAliveCells(unsigned char *matrix, size_t x0, size_t x1, size_t x2, size_t y0, size_t y1, size_t y2) {
    return matrix[x0 + y0] + matrix[x1 + y0] + matrix[x2 + y0] + matrix[x0 + y1]
         + matrix[x2 + y1] + matrix[x0 + y2] + matrix[x1 + y2] + matrix[x2 + y2];
}*/

void swap(void **a, void **b);
void kernel_compute_gen(unsigned char *matrix_src, unsigned char *matrix_dst,  unsigned int rows, unsigned int cols,long int iterations);
long int getDeviceInfo();
void compute_generation_on_gpu(GenStateGpu_p s1, GenStateGpu_p s2, int iterations);
GenStateGpu_p create_gen_gpu(int rows, int cols);
void gen_h2d(GenState_p gen_host, GenStateGpu_p gen_device);
void gen_d2h(GenState_p gen_host, GenStateGpu_p gen_device);
//void random_gen_gpu(GenStateGpu_p gen);
void free_gen_gpu(GenStateGpu_p gen);
void game_Gpu(int rows, int cols, int iterations);
