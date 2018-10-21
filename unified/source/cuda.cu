#include <cuda.h>

/*
__device__ uint8_t countAliveCells(uint8_t *matrix, uint32_t x0, uint32_t x1, uint32_t x2, uint32_t y0, uint32_t y1, uint32_t y2) {
    return matrix[x0 + y0] + matrix[x1 + y0] + matrix[x2 + y0] + matrix[x0 + y1]
         + matrix[x2 + y1] + matrix[x0 + y2] + matrix[x1 + y2] + matrix[x2 + y2];
}*/

__device__ void gpu_swap(void **a, void **b) {
    void *temp = *a;
    *a = *b;
    *b = temp;
}

/**
 * Kernel device routine
 */
__global__ void kernel_compute_gen(uint8_t *matrix_src, uint8_t *matrix_dst,  uint32_t rows, uint32_t cols, uint32_t iterations) {
    //each thread loads one element from global to shared mem
    uint32_t i = threadIdx.x;
    uint32_t dim = rows * cols;
    
    uint32_t x1 = i & cols-1;//% cols;
    uint32_t x0 = (x1 - 1) & cols-1;//% cols;
    uint32_t x2 = (x1 + 1) & cols-1;//% cols;

    uint32_t y1 = i - x1;
    uint32_t y0 = (y1 - cols) & dim-1;//% dim;
    uint32_t y2 = (y1 + cols) & dim-1;//% dim;

    for (uint32_t i = 0; i < iterations; i++){
        uint8_t aliveCells = matrix_src[x0 + y0] + matrix_src[x1 + y0] + matrix_src[x2 + y0] + matrix_src[x0 + y1] +
                                    matrix_src[x2 + y1] + matrix_src[x0 + y2] + matrix_src[x1 + y2] + matrix_src[x2 + y2];
        matrix_dst[y1 + x1] = (aliveCells == 3 || (aliveCells == 2 && matrix_src[y1 + x1])) ? 1 : 0; 
        gpu_swap((void**)&matrix_dst,(void**)&matrix_src);
        __syncthreads();
    } 
}

uint32_t getDeviceInfo(){
    int deviceCount = 1;
    cudaSetDevice(deviceCount);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceCount);
    return deviceProp.maxThreadsPerBlock;
}

void compute_generation_on_gpu(GenStateGpu_p s1, GenStateGpu_p s2, uint32_t iterations){
    // Load gpu info for optimal load balancing   
    uint32_t dim_world = s1->rows * s1->cols;
    uint32_t threadsPerBlock = getDeviceInfo();

    if (dim_world <= threadsPerBlock){
        kernel_compute_gen<<<1, dim_world>>>(s1->matrix, s2->matrix, s1->rows, s1->cols, iterations);//num_block, dim_block,  
    } /*else {
        uint32_t totalThreads = world_size / 32; 
        if ( totalThreads <= threadsPerBlock){
            kernel con un unico blocco con la barrier interna
        } else {
            uint32_t totalBlocks = totalThreads / threadsPerBlock; 
            for (uint32_t i = 0; i< iterations; i++)
            //swap((void *) &s1, (void *) &s2);
        }
    }*/
}

uint8_t isPow2(uint32_t x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}

GenStateGpu_p create_gen_gpu(uint32_t rows, uint32_t cols){
    uint32_t size = rows * cols * sizeof(uint8_t);
    uint8_t *matrix = NULL; 
    cudaError_t err = cudaMalloc(&matrix, size);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to allocate d_Input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    GenStateGpu_p gen = (GenStateGpu_p) malloc(sizeof(GenStateGpu));
    gen->matrix = matrix;
    gen->rows = rows;
    gen->cols = cols;
    return gen;
}

void gen_h2d(GenState_p gen_host, GenStateGpu_p gen_device){
    if (gen_host->rows == gen_device->rows && gen_host->cols == gen_device->cols){
        cudaError_t err = cudaMemcpy(gen_device->matrix, gen_host->matrix, sizeof(uint8_t)*gen_host->rows * gen_host->cols, cudaMemcpyHostToDevice);//dest, sorg, size, kind
        
        if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy GenState from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        }
    }
}

void gen_d2h(GenState_p gen_host, GenStateGpu_p gen_device){
    if (gen_host->rows == gen_device->rows && gen_host->cols == gen_device->cols){
        cudaError_t err = cudaMemcpy(gen_host->matrix, gen_device->matrix, sizeof(uint8_t)*gen_host->rows * gen_host->cols, cudaMemcpyDeviceToHost);//dest, sorg, size, kind
        
        if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy GenState device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        }
    }
}

/*void random_gen_gpu(GenStateGpu_p gen){
    for (uint32_t i=0; i < gen->cols * gen->rows; i++){
        gen->matrix[i] = rand() & 1; 
    }
}*/

void random_gen(GenState_p gen){
    for (uint32_t i=0; i < gen->cols * gen->rows; i++){
        gen->matrix[i] = rand() & 1; 
    }
}

void free_gen_gpu(GenStateGpu_p gen){
    cudaFree(gen->matrix);
    free(gen);
}

GenState_p create_gen(uint32_t rows, uint32_t cols){
    GenState_p gen = (GenState_p) malloc(sizeof(GenState));
    gen->matrix = (uint8_t *) malloc(cols * rows * sizeof(uint8_t));
    gen->rows = rows;
    gen->cols = cols;
    return gen;
}

void free_gen(GenState_p gen){
    free(gen->matrix);
    free(gen);
}

void game_Gpu(uint32_t rows, uint32_t cols, uint32_t iterations){
    if (isPow2(rows) && isPow2(rows)){
        GenState_p gen_host = create_gen(rows, cols);
        random_gen(gen_host);
        GenStateGpu_p gen_device_1 = create_gen_gpu(rows, cols);
        GenStateGpu_p gen_device_2 = create_gen_gpu(rows, cols);
        gen_h2d(gen_host, gen_device_1);
        compute_generation_on_gpu(gen_device_1, gen_device_2, iterations);
        
        // FINE KERNEL
        free_gen(gen_host);
        free_gen_gpu(gen_device_1);
        free_gen_gpu(gen_device_2);
    } else {
        puts("Rows or Cols are not a power of 2!");
    }
}

/*
 * Host main routine
 */
int main(void){
    game_Gpu(32,32, 1000);
    printf("eseguito cuda");
}
