#include <cudax.h>

#define MAX_CELL_PER_THREAD 1

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
 * Kernel 1: Tutto il mondo in un solo blocco con 1 cella per thread
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
 * Kernel 2: Tutto il mondo in un solo blocco con piu' celle per thread
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

/*
*  kernel 3: Il mondo su piu' blocchi con una cella per thread 
*/
__global__ void kernel_compute_gen_multiblocks(uint8_t *matrix_src, uint8_t *matrix_dst,  uint32_t dim_m1, uint32_t cols) {
    uint32_t cell = blockIdx.x*blockDim.x + threadIdx.x;
    //int row = blockIdx.y * blockDim.y + threadIdx.y;
    //int col = blockIdx.x * blockDim.x + threadIdx.x; 

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

/*
*  kernel 3,5: Il mondo su piu' blocchi, acceduto come matrice, con una cella per thread 
*/
__global__ void kernel_compute_gen_multiblocks_multidim(uint8_t *matrix_src, uint8_t *matrix_dst) {
    uint8_t (*msrc)[MULTIDIM_R][MULTIDIM_C] = (uint8_t (*)[MULTIDIM_R][MULTIDIM_C]) matrix_src; 
    uint8_t (*mdst)[MULTIDIM_R][MULTIDIM_C] = (uint8_t (*)[MULTIDIM_R][MULTIDIM_C]) matrix_dst;

    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; 
    uint32_t col = blockIdx.y * blockDim.y + threadIdx.y; 
    
    uint8_t aliveCells =  (*msrc)[(row - 1) & MULTIDIM_R-1][(col - 1) & MULTIDIM_C-1];     
            aliveCells += (*msrc)[ row]                    [(col - 1) & MULTIDIM_C-1]; 
            aliveCells += (*msrc)[(row + 1) & MULTIDIM_R-1][(col - 1) & MULTIDIM_C-1];
            
            aliveCells += (*msrc)[(row - 1) & MULTIDIM_R-1][col]; 
            aliveCells += (*msrc)[(row + 1) & MULTIDIM_R-1][col]; 
            
            aliveCells += (*msrc)[(row - 1) & MULTIDIM_R-1][(col + 1) & MULTIDIM_C-1]; 
            aliveCells += (*msrc)[ row]                    [(col + 1) & MULTIDIM_C-1]; 
            aliveCells += (*msrc)[(row + 1) & MULTIDIM_R-1][(col + 1) & MULTIDIM_C-1];

    
    (*mdst)[row][col] = (aliveCells == 3 || (aliveCells == 2 && (*msrc)[row][col])) ? 1 : 0;
}

/*
*  kernel 4: Il mondo su piu' blocchi con una cella per thread e shared memory dinamica. 
*            Numero di thread uguale al numero di celle del mondo, con piu' copie per thread
*/
__global__ void kernel_compute_gen_shared(uint8_t *matrix_src, uint8_t *matrix_dst,  uint32_t dim_m1, uint32_t cols) {
    extern __shared__ uint8_t shared[];
    uint32_t cell = blockIdx.x*blockDim.x + threadIdx.x;

    uint32_t x1 = cell     & cols-1; //% cols;
    uint32_t x0 = (x1 - 1) & cols-1; //% cols;
    uint32_t x2 = (x1 + 1) & cols-1; //% cols;

    uint32_t y1 = cell - x1;
    uint32_t y0 = (y1 - cols) & dim_m1; //% dim;
    uint32_t y2 = (y1 + cols) & dim_m1; //% dim;
    
    //non aumentiamo il numero di thread durante la copia perche' dovremmo sprecare warp_size thread in quel caso
    shared[threadIdx.x + 1]                         = matrix_src[x1 + y0];
    shared[threadIdx.x + 1 + blockDim.x + 2]        = matrix_src[cell];
    shared[threadIdx.x + 1 + (blockDim.x + 2) * 2]  = matrix_src[x1 + y2];
    
    
    if (threadIdx.x == 0){
        shared[0]                     = matrix_src[x0 + y0];
        shared[blockDim.x + 2]        = matrix_src[x0 + y1];
        shared[(blockDim.x + 2) * 2]  = matrix_src[x0 + y2];
    } else if (threadIdx.x == blockDim.x-1){
        shared[blockDim.x + 2 - 1]        = matrix_src[x2 + y0];
        shared[(blockDim.x + 2) * 2 - 1]  = matrix_src[x2 + y1];
        shared[(blockDim.x + 2) * 3 - 1]  = matrix_src[x2 + y2];
    }

    __syncthreads();

    uint32_t scell = threadIdx.x + 1 + blockDim.x + 2;

    uint32_t sx1 = scell;//- blockDim.x - 2;    //& cols-1; //% cols;
    uint32_t sx0 = (sx1 - 1); //& cols-1; //% cols;
    uint32_t sx2 = (sx1 + 1); //& cols-1; //% cols;

    uint32_t sy1 = scell - sx1;
    uint32_t sy0 = (sy1 - blockDim.x - 2); // cosi sale di una riga //& dim_m1; //% dim;
    uint32_t sy2 = (sy1 + blockDim.x + 2); //Scende di una riga //& dim_m1; //% dim;

    uint8_t aliveCells = shared[sx0 + sy0] + shared[sx1 + sy0] + shared[sx2 + sy0] + shared[sx0 + sy1] +
                         shared[sx2 + sy1] + shared[sx0 + sy2] + shared[sx1 + sy2] + shared[sx2 + sy2];

    matrix_dst[y1 + x1] = (aliveCells == 3 || (aliveCells == 2 && shared[scell])) ? 1 : 0;      
}


/*
*  kernel 5: Il mondo su piu' blocchi con una cella per thread e shared memory statica (per accederla come matrice). 
*            Numero di thread maggiore del numero di celle del mondo, per effettuare solo una copia per thread
*/
__global__ void kernel_compute_gen_last_shared(uint8_t *matrix_src, uint8_t *matrix_dst,  uint32_t rows, uint32_t cols) {
    __shared__ int shared[3][128 + 2];

    int ix = ((blockDim.x - 2) * blockIdx.x + threadIdx.x) & (cols - 1);
    int iy = (blockIdx.y + threadIdx.y) & (rows - 1);
    int id = iy * cols + ix;

    int i = threadIdx.y;
    int j = threadIdx.x;

    uint8_t mine = matrix_src[id]; // keep cell in register
    shared[i][j] = mine;
    //shared[i][j] = matrix_src[id];
 
    __syncthreads();

    if (i == 1 && j > 0 && j < 129){

        uint8_t aliveCells = shared[i + 1][j] +  // lower
                             shared[i - 1][j] +  // upper
                             shared[i][j + 1] +  // right
                             shared[i][j - 1] +  // left
                             shared[i + 1][j + 1] + 
                             shared[i - 1][j - 1] +  //diagonals
                             shared[i - 1][j + 1] + 
                             shared[i + 1][j - 1];

        matrix_dst[id] = (aliveCells == 3 || (aliveCells == 2 && mine)) ? 1 : 0;
    }          
}



int getMultiprocessorCores(cudaDeviceProp devProp){  
    int cores = 0;
    switch (devProp.major){
        case 2: // Fermi
            if (devProp.minor == 1) cores = 48;
            else cores = 32;
        break;
        case 3: // Kepler
            cores = 192;
        break;
        case 5: // Maxwell
            cores = 128;
        break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = 128;
            else if (devProp.minor == 0) cores = 64;
            else printf("Unknown device type\n");
        break;
        case 7: // Volta
            if (devProp.minor == 0) cores = 64;
            else printf("Unknown device type\n");
        break;
        default:
            printf("Unknown device type\n"); 
        break;
    }
    return cores;
}

// Restutuisce il max tra numero di core per stream multiprocessor e numero di thread massimi per blocco
uint32_t getDeviceInfo(){
    uint16_t deviceCount = 0;
    cudaSetDevice(deviceCount);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceCount);
    uint32_t threadPerBlock = deviceProp.maxThreadsPerBlock;
    uint32_t multiprocessorCores = getMultiprocessorCores(deviceProp); 
    return (threadPerBlock <= multiprocessorCores) ? threadPerBlock : multiprocessorCores;  
}


// Richiama il kernel piu' corretto in base alla dimensione del mondo
void compute_generation_on_gpu(GenStateGpu_p s1, GenStateGpu_p s2, uint32_t iterations){
    // Load gpu info for optimal load balancing   
    uint32_t dim_world = s1->rows * s1->cols;
    uint32_t threadsPerBlock = getDeviceInfo();

    uint32_t totalBlocks = dim_world / threadsPerBlock;//the number of blocks will always be a positive power of two 
    if (dim_world <= threadsPerBlock){
        kernel_compute_gen_singleblock_1<<<1, dim_world>>>(s1->matrix, s2->matrix,  dim_world-1, s1->cols, iterations);//num_block, dim_block,  
    } else {
        for (uint32_t iter = 0; iter< iterations; iter++){
            kernel_compute_gen_multiblocks<<<totalBlocks, threadsPerBlock>>>(s1->matrix, s2->matrix, dim_world-1, s1->cols);//num_block, dim_block,          
            swap((void **) &s1, (void **) &s2);  
        }
    }    
}

// funzione di appoggio
void compute_cpu_generations_on_gpu(GenState_p s1, GenState_p s2, uint32_t iterations){
    GenStateGpu_p gen_device_1 = create_gen_gpu(s1->rows, s1->cols);
    GenStateGpu_p gen_device_2 = create_gen_gpu(s1->rows, s1->cols);

    gen_h2d(s1, gen_device_1);
    compute_generation_on_gpu(gen_device_1, gen_device_2, iterations);
    gen_d2h(gen_device_2, s2);
    
    free_gen_gpu(gen_device_1);
    free_gen_gpu(gen_device_2);
}


// Richiama il kernel piu' corretto in base alla dimensione del mondo
void compute_generation_on_gpu_multidim(GenStateGpu_p s1, GenStateGpu_p s2, uint32_t iterations){
    // Load gpu info for optimal load balancing   
    uint32_t dim_world = s1->cols * s1->rows;
    uint32_t threadsPerBlock = getDeviceInfo();

    if (dim_world <= threadsPerBlock){
        kernel_compute_gen_singleblock_1<<<1, dim_world>>>(s1->matrix, s2->matrix,  dim_world-1, s1->cols, iterations);//num_block, dim_block,  
    } else {
        //printf("totalBlocks: %d\n",totalBlocks);
        dim3 dimBlock(16, 8);
        dim3 dimGrid(s1->cols/16, s1->rows/8);
        for (uint32_t iter = 0; iter< iterations; iter++){
            kernel_compute_gen_multiblocks_multidim<<<dimGrid, dimBlock>>>(s1->matrix, s2->matrix);//num_block, dim_block,          
            swap((void **) &s1, (void **) &s2);  
        }
    }    
}

// funzione di appoggio
void compute_cpu_generations_on_gpu_multidim(GenState_p s1, GenState_p s2, uint32_t iterations){
    GenStateGpu_p gen_device_1 = create_gen_gpu(s1->rows, s1->cols);
    GenStateGpu_p gen_device_2 = create_gen_gpu(s1->rows, s1->cols);

    gen_h2d(s1, gen_device_1);
    compute_generation_on_gpu_multidim(gen_device_1, gen_device_2, iterations);
    gen_d2h(gen_device_2, s2);
    
    free_gen_gpu(gen_device_1);
    free_gen_gpu(gen_device_2);
}


// Richiama il kernel piu' corretto in base alla dimensione del mondo
void compute_generation_on_gpu_shared(GenStateGpu_p s1, GenStateGpu_p s2, uint32_t iterations){
    // Load gpu info for optimal load balancing   
    uint32_t dim_world = s1->rows * s1->cols;
    uint32_t threadsPerBlock = getDeviceInfo();

    uint32_t totalBlocks = dim_world / threadsPerBlock;//the number of blocks will always be a positive power of two 
    if (dim_world <= threadsPerBlock){
        kernel_compute_gen_singleblock_1<<<1, dim_world>>>(s1->matrix, s2->matrix,  dim_world-1, s1->cols, iterations);//num_block, dim_block,  
    } else {
        if (s1->cols >= threadsPerBlock){
            cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
            for (uint32_t iter = 0; iter< iterations; iter++){
                //kernel_compute_gen_shared<<<totalBlocks, threadsPerBlock, sizeof(uint8_t)*((threadsPerBlock+2)*3)>>>(s1->matrix, s2->matrix, dim_world-1, s1->cols);
                dim3 dimBlock(threadsPerBlock + 2, 3);
                dim3 dimGrid(s1->cols/threadsPerBlock, s1->rows);
                kernel_compute_gen_last_shared<<<dimGrid, dimBlock>>>(s1->matrix, s2->matrix, s1->rows, s1->cols);//num_block, dim_block,          
                swap((void **) &s1, (void **) &s2);  
            }
        } else { // per tutti i casi in cui le colonne NON sono multiple del numero di core per SMP (dato che vengono usati slice di questa dimensione) 
            for (uint32_t iter = 0; iter< iterations; iter++){                
                kernel_compute_gen_multiblocks<<<totalBlocks, threadsPerBlock>>>(s1->matrix, s2->matrix, dim_world-1, s1->cols);//num_block, dim_block,          
                swap((void **) &s1, (void **) &s2);  
            }
        }
    }    
}

// funzione di appoggio

void compute_cpu_generations_on_gpu_shared(GenState_p s1, GenState_p s2, uint32_t iterations){
    GenStateGpu_p gen_device_1 = create_gen_gpu(s1->rows, s1->cols);
    GenStateGpu_p gen_device_2 = create_gen_gpu(s1->rows, s1->cols);

    gen_h2d(s1, gen_device_1);
    compute_generation_on_gpu_shared(gen_device_1, gen_device_2, iterations);
    gen_d2h(gen_device_2, s2);
    
    free_gen_gpu(gen_device_1);
    free_gen_gpu(gen_device_2);
}
