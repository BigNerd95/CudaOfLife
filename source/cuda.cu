#include <cudax.h>

#define MAX_CELL_PER_THREAD 2

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

void gen_d2h(GenStateGpu_p gen_device, GenState_p gen_host){
    if (gen_host->rows == gen_device->rows && gen_host->cols == gen_device->cols){
        cudaError_t err = cudaMemcpy(gen_host->matrix, gen_device->matrix, sizeof(uint8_t)*gen_host->rows * gen_host->cols, cudaMemcpyDeviceToHost);//dest, sorg, size, kind
        
        if (err != cudaSuccess){
            fprintf(stderr, "Failed to copy GenState device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
}

void free_gen_gpu(GenStateGpu_p gen){
    cudaFree(gen->matrix);
    free(gen);
}

__device__ void gpu_swap(void **a, void **b) {
    void *temp = *a;
    *a = *b;
    *b = temp;
}

/**
 * 1) Kernel device routine
 */
__global__ void kernel_compute_gen_singleblock_1(uint8_t *matrix_src, uint8_t *matrix_dst,  uint32_t dim_m1, uint32_t cols, uint32_t iterations) {
    uint32_t cell = threadIdx.x;
    
    uint32_t x1 = cell     & cols-1; //% cols;
    uint32_t x0 = (x1 - 1) & cols-1; //% cols;
    uint32_t x2 = (x1 + 1) & cols-1; //% cols;

    uint32_t y1 = cell - x1;
    uint32_t y0 = (y1 - cols) & dim_m1; //% dim;
    uint32_t y2 = (y1 + cols) & dim_m1; //% dim;

    for (uint32_t iter = 0; iter < iterations; iter++){
        uint8_t aliveCells = matrix_src[x0 + y0] + matrix_src[x1 + y0] + matrix_src[x2 + y0] + matrix_src[x0 + y1] +
                                    matrix_src[x2 + y1] + matrix_src[x0 + y2] + matrix_src[x1 + y2] + matrix_src[x2 + y2];
        matrix_dst[y1 + x1] = (aliveCells == 3 || (aliveCells == 2 && matrix_src[y1 + x1])) ? 1 : 0; 
        gpu_swap((void**)&matrix_dst,(void**)&matrix_src);
        __syncthreads();
    } 
}

/**
 * 2) Kernel device routine
 */
 __global__ void kernel_compute_gen_singleblock(uint8_t *matrix_src, uint8_t *matrix_dst,  uint32_t dim_m1, uint32_t cols, uint32_t iterations, uint32_t cellPerThreads) {
    uint32_t start = threadIdx.x * cellPerThreads;
    uint32_t end =  cellPerThreads + start;

    for (uint32_t iter = 0; iter < iterations; iter++){
        for (uint32_t cell = start; cell < end; cell++){
            uint32_t x1 = cell     & cols-1; //% cols;
            uint32_t x0 = (x1 - 1) & cols-1; //% cols;
            uint32_t x2 = (x1 + 1) & cols-1; //% cols;

            uint32_t y1 = cell - x1;
            uint32_t y0 = (y1 - cols) & dim_m1; //% dim;
            uint32_t y2 = (y1 + cols) & dim_m1; //% dim;

            uint8_t aliveCells = matrix_src[x0 + y0] + matrix_src[x1 + y0] + matrix_src[x2 + y0] + matrix_src[x0 + y1] +
                                 matrix_src[x2 + y1] + matrix_src[x0 + y2] + matrix_src[x1 + y2] + matrix_src[x2 + y2];
        
            matrix_dst[y1 + x1] = (aliveCells == 3 || (aliveCells == 2 && matrix_src[y1 + x1])) ? 1 : 0; 
        }

        gpu_swap((void**)&matrix_dst,(void**)&matrix_src);
        __syncthreads();
    } 
}

/**
 * 3) Kernel device routine
 */
 __global__ void kernel_compute_gen_multiblocks(uint8_t *matrix_src, uint8_t *matrix_dst,  uint32_t dim_m1, uint32_t cols, uint32_t cellPerThreads) {
    uint32_t start = (blockIdx.x*blockDim.x + threadIdx.x) * cellPerThreads;
    uint32_t end =  cellPerThreads + start;

    for (uint32_t cell = start; cell < end; cell++){
        uint32_t x1 = cell     & cols-1; //% cols;
        uint32_t x0 = (x1 - 1) & cols-1; //% cols;
        uint32_t x2 = (x1 + 1) & cols-1; //% cols;

        uint32_t y1 = cell - x1;
        uint32_t y0 = (y1 - cols) & dim_m1; //% dim;
        uint32_t y2 = (y1 + cols) & dim_m1; //% dim;

        uint8_t aliveCells = matrix_src[x0 + y0] + matrix_src[x1 + y0] + matrix_src[x2 + y0] + matrix_src[x0 + y1] +
                                matrix_src[x2 + y1] + matrix_src[x0 + y2] + matrix_src[x1 + y2] + matrix_src[x2 + y2];
    
        matrix_dst[y1 + x1] = (aliveCells == 3 || (aliveCells == 2 && matrix_src[y1 + x1])) ? 1 : 0; 
    }     
}

uint32_t getDeviceInfo(){
    int deviceCount = 0;
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
        //printf("\n1 if\n");
        kernel_compute_gen_singleblock_1<<<1, dim_world>>>(s1->matrix, s2->matrix,  dim_world-1, s1->cols, iterations);//num_block, dim_block,  
    } else {

        uint32_t cellPerThreads = dim_world / threadsPerBlock; 
        if (cellPerThreads <= MAX_CELL_PER_THREAD){ 
            //printf("\n2 if\n");
            kernel_compute_gen_singleblock<<<1, threadsPerBlock>>>(s1->matrix, s2->matrix, dim_world-1, s1->cols, iterations, cellPerThreads);//num_block, dim_block,          
        } else {//if there are too cells per thread we use more blocks to spread the works
            
            //printf("\n3 if\n");
            uint32_t totalBlocks = cellPerThreads / MAX_CELL_PER_THREAD;//the number of blocks will always be a positive power of two 
            uint32_t dim_m1 = dim_world-1;
            for (uint32_t iter = 0; iter< iterations; iter++){
                kernel_compute_gen_multiblocks<<<totalBlocks, threadsPerBlock>>>(s1->matrix, s2->matrix, dim_m1, s1->cols, MAX_CELL_PER_THREAD);//num_block, dim_block,          
                swap((void **) &s1, (void **) &s2);  
            }
        }
    }
}
 
void compute_cpu_generations_on_gpu(GenState_p s1, GenState_p s2, uint32_t iterations){
    GenStateGpu_p gen_device_1 = create_gen_gpu(s1->rows, s1->cols);
    GenStateGpu_p gen_device_2 = create_gen_gpu(s1->rows, s1->cols);

    gen_h2d(s1, gen_device_1);
    compute_generation_on_gpu(gen_device_1, gen_device_2, iterations);
    gen_d2h(gen_device_2, s2);
    
    free_gen_gpu(gen_device_1);
    free_gen_gpu(gen_device_2);
}



