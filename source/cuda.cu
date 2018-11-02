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

/*
Kenel 4 con 1 cell per threads
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
Kenel 4,5 matrice multidimensionale
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
Kenel 5 SHARED
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
    
    /*if (threadIdx.x < 3){
        uint32_t start_col = cell % cols;
        uint32_t first = ((((blockIdx.x * blockDim.x -1 ) & (cols - 1)) +  ) - cols ) & dim_m1; 
        shared[(blockDim.x + 2) * threadIdx.x] = matrix_src[(first + cols * threadIdx.x) & dim_m1];
    } else if (threadIdx.x < 6){

    }*/

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
Kernel 6 final fantasy 
*/
__global__ void kernel_compute_gen_last_shared(uint8_t *matrix_src, uint8_t *matrix_dst,  uint32_t dim_m1, uint32_t cols) {
    extern __shared__ uint8_t shared[];
    uint32_t cell = blockIdx.x*blockDim.x + threadIdx.x;// + ((threadIdx.y - 1) * cols));// & dim_m1;
    
    
    uint32_t x1 = cell & cols-1; //modlulo mondo esterno
    uint32_t y1 = (cell - x1) & dim_m1;

    if (threadIdx.x == 0 && blockIdx.x == 2){
        printf("\nx1: %d", x1);
        printf("\ny1: %d", y1);    
    }
    //uint32_t x0 = (x1 - 1) & cols-1; //% cols;
    //uint32_t x2 = (x1 + 1) & cols-1; //% cols;
    //uint32_t y0 = (y1 - cols) & dim_m1; //% dim;
    //uint32_t y2 = (y1 + cols) & dim_m1; //% dim;
    

    shared[threadIdx.x] = matrix_src[x1 + y1];
    
    __syncthreads();

    //if (threadIdx.x == 0 && blockIdx.x > 8000)
      //  printf("blockid.x: %d\n", blockIdx.x);

    if (threadIdx.x - 131 < 128){
        uint32_t scell = threadIdx.x;
        //if (threadIdx.x == 259)
          //  printf("Block dim: %d\n", blockDim.x);

        uint8_t aliveCells = shared[scell - 128 - 3] + //sx0 + sy0 
                             shared[scell - 128 - 2] + //sx1 + sy0
                             shared[scell - 128 - 1] + //sx2 + sy0 
                             shared[scell - 1] + //sx0 + sy1
                             shared[scell + 1] + //sx2 + sy1
                             shared[scell + 128 + 3] + //sx0 + sy2 
                             shared[scell + 128 + 2] + //sx1 + sy2 
                             shared[scell + 128 + 1]; //sx2 + sy2

        
        matrix_dst[x1 + y1] = (aliveCells == 3 || (aliveCells == 2 && shared[scell])) ? 1 : 0;
    }          
}


/*
#####################

__global__ void ghostRows(int dim, int* grid)
{
    // We want id ∈ [1,dim]
    int id = blockDim.x * blockIdx.x + threadIdx.x + 1;
 
    if (id <= dim)
    {
        //Copy first real row to bottom ghost row
        grid[(dim+2)*(dim+1)+id] = grid[(dim+2)+id];
        //Copy last real row to top ghost row
        grid[id] = grid[(dim+2)*dim + id];
    }
}
__global__ void ghostCols(int dim, int* grid)
{
    // We want id ∈ [0,dim+1]
    int id = blockDim.x * blockIdx.x + threadIdx.x;
 
    if (id <= dim+1)
    {
        //Copy first real column to right most ghost column
        grid[id*(dim+2)+dim+1] = grid[id*(dim+2)+1];
        //Copy last real column to left most ghost column
        grid[id*(dim+2)] = grid[id*(dim+2) + dim];                                                 
    }
}
 
__global__ void GOL(int dim, int *grid, int *newGrid)
{
        int iy = (blockDim.y -2) * blockIdx.y + threadIdx.y;
        int ix = (blockDim.x -2) * blockIdx.x + threadIdx.x;
        int id = iy * (dim+2) + ix;
 
        int i = threadIdx.y;
        int j = threadIdx.x;
        int numNeighbors;
 
        // Declare the shared memory on a per block level
        __shared__ int s_grid[BLOCK_SIZE_y][BLOCK_SIZE_x];
 
       // Copy cells into shared memory
       if (ix <= dim+1 && iy <= dim+1)
           s_grid[i][j] = grid[id];
 
       //Sync all threads in block
        __syncthreads();
 
       if (iy <= dim && ix <= dim) {
           if(i != 0 && i !=blockDim.y-1 && j != 0 && j !=blockDim.x-1) {
 
               // Get the number of neighbors for a given grid point
               numNeighbors = s_grid[i+1][j] + s_grid[i-1][j] //upper lower
                            + s_grid[i][j+1] + s_grid[i][j-1] //right left
                            + s_grid[i+1][j+1] + s_grid[i-1][j-1] //diagonals
                            + s_grid[i-1][j+1] + s_grid[i+1][j-1];
 
                int cell = s_grid[i][j];
 
                // Here we have explicitly all of the game rules
                if (cell == 1 && numNeighbors < 2)
                    newGrid[id] = 0;
                else if (cell == 1 && (numNeighbors == 2 || numNeighbors == 3))
                    newGrid[id] = 1;
                else if (cell == 1 && numNeighbors > 3)
                    newGrid[id] = 0;
                else if (cell == 0 && numNeighbors == 3)
                    newGrid[id] = 1;
                else
                    newGrid[id] = cell;
           }
       }
}

void test(){
    int i,j,iter;
    int *h_grid; //Grid on host
    int *d_grid; //Grid on device
    int *d_newGrid; //Second grid used on device only
    int *d_tmpGrid; //tmp grid pointer used to switch between grid and newGrid
 
    int dim = 1024; //Linear dimension of our grid - not counting ghost cells
    int maxIter = 1<<10; //Number of game steps
 
    size_t bytes = sizeof(int)*(dim+2)*(dim+2);
    // Allocate host Grid used for initial setup and read back from device
    h_grid = (int*)malloc(bytes);
 
    // Allocate device grids
    cudaMalloc(&d_grid, bytes);
    cudaMalloc(&d_newGrid, bytes);
 
    // Assign initial population randomly
    srand(SRAND_VALUE);
    for(i = 1; i<=dim; i++) {
        for(j = 1; j<=dim; j++) {
            h_grid[i*(dim+2)+j] = rand() % 2;
        }
    }
 
    cudaFuncSetCacheConfig(GOL, cudaFuncCachePreferShared);
 
    // Copy over initial game grid (Dim-1 threads)
    cudaMemcpy(d_grid, h_grid, bytes, cudaMemcpyHostToDevice);
 
    dim3 blockSize(BLOCK_SIZE_x, BLOCK_SIZE_y,1);
    int linGrid_x = (int)ceil(dim/(float)(BLOCK_SIZE_x-2));
    int linGrid_y = (int)ceil(dim/(float)(BLOCK_SIZE_y-2));
    dim3 gridSize(linGrid_x,linGrid_y,1);
 
    dim3 cpyBlockSize(BLOCK_SIZE_x,1,1);
    dim3 cpyGridRowsGridSize((int)ceil(dim/(float)cpyBlockSize.x),1,1);
    dim3 cpyGridColsGridSize((int)ceil((dim+2)/(float)cpyBlockSize.x),1,1);
 
    // Main game loop
    for (iter = 0; iter<maxIter; iter++) {
 
        ghostRows<<<cpyGridRowsGridSize, cpyBlockSize>>>(dim, d_grid);
        ghostCols<<<cpyGridColsGridSize, cpyBlockSize>>>(dim, d_grid);
        GOL<<<gridSize, blockSize>>>(dim, d_grid, d_newGrid);
 
        // Swap our grids and iterate again
        d_tmpGrid = d_grid;
        d_grid = d_newGrid;
        d_newGrid = d_tmpGrid;
    }//iter loop
 
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
        printf("CUDA error %s\n",cudaGetErrorString(error));
 
    // Copy back results and sum
    cudaMemcpy(h_grid, d_grid, bytes, cudaMemcpyDeviceToHost);
 
    // Sum up alive cells and print results
    int total = 0;
    for (i = 1; i<=dim; i++) {
        for (j = 1; j<=dim; j++) {
            total += h_grid[i*(dim+2)+j];
        }
    }
    printf("Total Alive: %d\n", total);
 
    cudaFree(d_grid);
    cudaFree(d_newGrid);
    free(h_grid);
}

#####################
*/


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

uint32_t getDeviceInfo(){
    uint16_t deviceCount = 0;
    cudaSetDevice(deviceCount);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceCount);
    uint32_t threadPerBlock = deviceProp.maxThreadsPerBlock;
    uint32_t multiprocessorCores = getMultiprocessorCores(deviceProp); 
    return (threadPerBlock <= multiprocessorCores) ? threadPerBlock : multiprocessorCores;  
}


 
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

void compute_cpu_generations_on_gpu(GenState_p s1, GenState_p s2, uint32_t iterations){
    GenStateGpu_p gen_device_1 = create_gen_gpu(s1->rows, s1->cols);
    GenStateGpu_p gen_device_2 = create_gen_gpu(s1->rows, s1->cols);

    gen_h2d(s1, gen_device_1);
    compute_generation_on_gpu(gen_device_1, gen_device_2, iterations);
    gen_d2h(gen_device_2, s2);
    
    free_gen_gpu(gen_device_1);
    free_gen_gpu(gen_device_2);
}


void compute_generation_on_gpu_multidim(GenStateGpu_p s1, GenStateGpu_p s2, uint32_t iterations){
    // Load gpu info for optimal load balancing   
    uint32_t dim_world = s1->cols * s1->rows;
    uint32_t threadsPerBlock = getDeviceInfo();

    //uint32_t totalBlocks = dim_world / threadsPerBlock;//the number of blocks will always be a positive power of two 
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

void compute_cpu_generations_on_gpu_multidim(GenState_p s1, GenState_p s2, uint32_t iterations){
    GenStateGpu_p gen_device_1 = create_gen_gpu(s1->rows, s1->cols);
    GenStateGpu_p gen_device_2 = create_gen_gpu(s1->rows, s1->cols);

    gen_h2d(s1, gen_device_1);
    compute_generation_on_gpu_multidim(gen_device_1, gen_device_2, iterations);
    gen_d2h(gen_device_2, s2);
    
    free_gen_gpu(gen_device_1);
    free_gen_gpu(gen_device_2);
}



void compute_generation_on_gpu_shared(GenStateGpu_p s1, GenStateGpu_p s2, uint32_t iterations){
    // Load gpu info for optimal load balancing   
    uint32_t dim_world = s1->rows * s1->cols;
    uint32_t threadsPerBlock = getDeviceInfo();

    uint32_t totalBlocks = dim_world / threadsPerBlock;//the number of blocks will always be a positive power of two 
    if (dim_world <= threadsPerBlock){
        kernel_compute_gen_singleblock_1<<<1, dim_world>>>(s1->matrix, s2->matrix,  dim_world-1, s1->cols, iterations);//num_block, dim_block,  
    } else {
        if (s1->cols >= threadsPerBlock){
            for (uint32_t iter = 0; iter< iterations; iter++){
                //kernel_compute_gen_shared<<<totalBlocks, threadsPerBlock, sizeof(uint8_t)*((threadsPerBlock+2)*3)>>>(s1->matrix, s2->matrix, dim_world-1, s1->cols);
                //dim3 dimBlock(threadsPerBlock + 2, 3);
                int dimBlock = 390;
                kernel_compute_gen_last_shared<<<totalBlocks, dimBlock, sizeof(uint8_t)*((threadsPerBlock+2)*3)>>>(s1->matrix, s2->matrix, dim_world-1, s1->cols);//num_block, dim_block,          
                swap((void **) &s1, (void **) &s2);  
            }
        } else {
            for (uint32_t iter = 0; iter< iterations; iter++){                
                kernel_compute_gen_multiblocks<<<totalBlocks, threadsPerBlock>>>(s1->matrix, s2->matrix, dim_world-1, s1->cols);//num_block, dim_block,          
                swap((void **) &s1, (void **) &s2);  
            }
        }
    }    
}

void compute_cpu_generations_on_gpu_shared(GenState_p s1, GenState_p s2, uint32_t iterations){
    GenStateGpu_p gen_device_1 = create_gen_gpu(s1->rows, s1->cols);
    GenStateGpu_p gen_device_2 = create_gen_gpu(s1->rows, s1->cols);

    gen_h2d(s1, gen_device_1);
    compute_generation_on_gpu_shared(gen_device_1, gen_device_2, iterations);
    gen_d2h(gen_device_2, s2);
    
    free_gen_gpu(gen_device_1);
    free_gen_gpu(gen_device_2);
}