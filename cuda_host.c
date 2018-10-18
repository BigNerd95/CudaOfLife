#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>


typedef struct {
    unsigned char *matrix;
    unsigned int rows;
    unsigned int cols;
} GenState, *GenState_p;


/**
 * Kernel device routine
 */
__global__ void kernel_code(int *d_Input,int *d_Output) {
    extern __shared__ int sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = d_Input[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads(); //sincronizza solo all' interno dello stesso blocco
    }

    // write result for this block to global mem
    if (tid == 0){
        d_Output[blockIdx.x] = sdata[0];
    } 
    
}

void compute_generation_on_gpu(GenState_p s1, GenState_p s2){
    // Load gpu info for optimal load balancing   
    deviceCount = 1;
    cudaSetDevice(deviceCount);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceCount);
    long int threadsPerBlock = deviceProp.maxThreadsPerBlock;
    long int totalThreads = 0;
    long int totalBlocks = 0;

    if (dim_word <= threadsPerBlock){
        kernel con una cella per thread
    } else {
        totalThreads = world_size / 32; 
        if ( totalThreads <= threadsPerBlock){
            kernel con un unico blocco con la barrier interna
        } else {
            totalBlocks = totalThreads / threadsPerBlock; 
            calcolare il numero totale di blocchi
        }
    }
}

int isPow2(unsigned int x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}

GenState_p create_gen(int rows, int cols){
    if (isPow2(word_size)){
        size_t size = rows * cols * sizeof(unsigned char);
        unsigned char *matrix = NULL; 
        err = cudaMalloc(&matrix, size);
        if (err != cudaSuccess){
            fprintf(stderr, "Failed to allocate d_Input (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        GenState_p gen = (GenState_p) malloc(sizeof(GenState));
        gen->matrix = *matrix;
        gen->rows = rows;
        gen->cols = cols;
        return gen;
    }   
}

void random_gen(GenState_p gen){
    inserire qui il kernel che randomizza la matrice che tanto è gia in gpu gen->matrix
}

void free_gen(GenState_p gen){
    cudafree(gen->matrix);
    free(gen);
}

/*
 * Host main routine
 */
int main(void){
    cudaError_t err = cudaSuccess;
    GenState_p s1 = create_gen(rows, cols);
    GenState_p s2 = create_gen(rows, cols);
    long int dim_word = rows * cols;
    random_gen(s1);

    for (int i = 0; i < iterations; i++) {
        compute_generation_doublefor(s1, s2);
        swap((void *) &s1, (void *) &s2);
    }
   
    compute_generation_on_gpu(GenState_p s1, GenState_p s2);
    
    
    /*
    // Copy the device result vector in device memory to the host result vector in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_Output, d_Output, sizeof(int)*1, cudaMemcpyDeviceToHost);//dest, sorg, size, kind
    
    if (err != cudaSuccess){
    fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }

    */

    
    // FINE KERNEL
    free_gen(s1);
    free_gen(s2);

}
