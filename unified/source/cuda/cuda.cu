#include <cuda.h>

typedef struct {
    unsigned char *matrix;
    unsigned int rows;
    unsigned int cols;
} GenStateGpu, *GenStateGpu_p;

typedef struct {
    unsigned char *matrix;
    unsigned int rows;
    unsigned int cols;
} GenState, *GenState_p;

/*
__device__ unsigned char countAliveCells(unsigned char *matrix, size_t x0, size_t x1, size_t x2, size_t y0, size_t y1, size_t y2) {
    return matrix[x0 + y0] + matrix[x1 + y0] + matrix[x2 + y0] + matrix[x0 + y1]
         + matrix[x2 + y1] + matrix[x0 + y2] + matrix[x1 + y2] + matrix[x2 + y2];
}*/

__device__ void swap(void **a, void **b) {
    void *temp = *a;
    *a = *b;
    *b = temp;
}

/**
 * Kernel device routine
 */
__global__ void kernel_compute_gen(unsigned char *matrix_src, unsigned char *matrix_dst,  unsigned int rows, unsigned int cols,long int iterations) {
    //each thread loads one element from global to shared mem
    unsigned int i = threadIdx.x;
    size_t dim = rows * cols;
    
    size_t x1 = i & cols-1;//% cols;
    size_t x0 = (x1 - 1) & cols-1;//% cols;
    size_t x2 = (x1 + 1) & cols-1;//% cols;

    size_t y1 = i - x1;
    size_t y0 = (y1 - cols) & dim-1;//% dim;
    size_t y2 = (y1 + cols) & dim-1;//% dim;

    for (int i = 0; i < iterations; i++){
        unsigned char aliveCells = matrix_src[x0 + y0] + matrix_src[x1 + y0] + matrix_src[x2 + y0] + matrix_src[x0 + y1] +
                                    matrix_src[x2 + y1] + matrix_src[x0 + y2] + matrix_src[x1 + y2] + matrix_src[x2 + y2];
        matrix_dst[y1 + x1] = (aliveCells == 3 || (aliveCells == 2 && matrix_src[y1 + x1])) ? 1 : 0; 
        swap((void**)&matrix_dst,(void**)&matrix_src);
        __syncthreads();
    } 
}

long int getDeviceInfo(){
    int deviceCount = 1;
    cudaSetDevice(deviceCount);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceCount);
    return deviceProp.maxThreadsPerBlock;
}

void compute_generation_on_gpu(GenStateGpu_p s1, GenStateGpu_p s2, int iterations){
    // Load gpu info for optimal load balancing   
    long int dim_world = s1->rows * s1->cols;
    long int threadsPerBlock = getDeviceInfo();

    if (dim_world <= threadsPerBlock){
        kernel_compute_gen<<<1, dim_world>>>(s1->matrix, s2->matrix, s1->rows, s1->cols, iterations);//num_block, dim_block,  
    } /*else {
        long int totalThreads = world_size / 32; 
        if ( totalThreads <= threadsPerBlock){
            kernel con un unico blocco con la barrier interna
        } else {
            long int totalBlocks = totalThreads / threadsPerBlock; 
            for (int i = 0; i< iterations; i++)
            //swap((void *) &s1, (void *) &s2);
        }
    }*/
}

int isPow2(unsigned int x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}

GenStateGpu_p create_gen_gpu(int rows, int cols){
    size_t size = rows * cols * sizeof(unsigned char);
    unsigned char *matrix = NULL; 
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
        cudaError_t err = cudaMemcpy(gen_device->matrix, gen_host->matrix, sizeof(unsigned char)*gen_host->rows * gen_host->cols, cudaMemcpyHostToDevice);//dest, sorg, size, kind
        
        if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy GenState from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        }
    }
}

void gen_d2h(GenState_p gen_host, GenStateGpu_p gen_device){
    if (gen_host->rows == gen_device->rows && gen_host->cols == gen_device->cols){
        cudaError_t err = cudaMemcpy(gen_host->matrix, gen_device->matrix, sizeof(unsigned char)*gen_host->rows * gen_host->cols, cudaMemcpyDeviceToHost);//dest, sorg, size, kind
        
        if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy GenState device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
        }
    }
}

/*void random_gen_gpu(GenStateGpu_p gen){
    for (int i=0; i < gen->cols * gen->rows; i++){
        gen->matrix[i] = rand() & 1; 
    }
}*/

void random_gen(GenState_p gen){
    for (int i=0; i < gen->cols * gen->rows; i++){
        gen->matrix[i] = rand() & 1; 
    }
}

void free_gen_gpu(GenStateGpu_p gen){
    cudaFree(gen->matrix);
    free(gen);
}

GenState_p create_gen(int rows, int cols){
    GenState_p gen = (GenState_p) malloc(sizeof(GenState));
    gen->matrix = (unsigned char *) malloc(cols * rows * sizeof(unsigned char));
    gen->rows = rows;
    gen->cols = cols;
    return gen;
}

void free_gen(GenState_p gen){
    free(gen->matrix);
    free(gen);
}

void game_Gpu(int rows, int cols, int iterations){
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
    prinf("eseguito cuda");
}
