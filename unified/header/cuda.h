#include <common.h> 

typedef struct {
    uint8_t *matrix;
    uint32_t rows;
    uint32_t cols;
} GenStateGpu, *GenStateGpu_p;

/*
__device__ uint8_t countAliveCells(uint8_t *matrix, uint32_t x0, uint32_t x1, uint32_t x2, uint32_t y0, uint32_t y1, uint32_t y2) {
    return matrix[x0 + y0] + matrix[x1 + y0] + matrix[x2 + y0] + matrix[x0 + y1]
         + matrix[x2 + y1] + matrix[x0 + y2] + matrix[x1 + y2] + matrix[x2 + y2];
}*/


__global__ void kernel_compute_gen(uint8_t *matrix_src, uint8_t *matrix_dst,  uint32_t rows, uint32_t cols, uint32_t iterations);
uint32_t getDeviceInfo();
void compute_generation_on_gpu(GenStateGpu_p s1, GenStateGpu_p s2, uint32_t iterations);
GenStateGpu_p create_gen_gpu(uint32_t rows, uint32_t cols);
void gen_h2d(GenState_p gen_host, GenStateGpu_p gen_device);
void gen_d2h(GenState_p gen_host, GenStateGpu_p gen_device);
//void random_gen_gpu(GenStateGpu_p gen);
void free_gen_gpu(GenStateGpu_p gen);
void game_Gpu(uint32_t rows, uint32_t cols, uint32_t iterations);
