#include <openmp.h> 

void omp_compute_generations(GenState_p s1, GenState_p s2, uint32_t iterations){
    uint32_t rows_m1 = s1->rows - 1;
    uint32_t cols_m1 = s1->cols - 1;
    uint32_t cols_p2 = log2pow2(s1->cols);
    
    for (uint32_t iter = 0; iter < iterations; iter ++){

        #pragma omp parallel for
        for (uint32_t y = 0; y < s1->rows; y++) {
            uint32_t y0 = ((y - 1) & rows_m1) << cols_p2; //* s1->cols;
            uint32_t y1 = y                   << cols_p2; //* s1->cols;
            uint32_t y2 = ((y + 1) & rows_m1) << cols_p2; //* s1->cols;

            for (uint32_t x = 0; x < s1->cols; x++) {
                uint32_t x0 = (x - 1) & cols_m1;
                uint32_t x2 = (x + 1) & cols_m1;

                uint8_t aliveCells = countAliveCells(s1->matrix, x0, x, x2, y0, y1, y2);
                s2->matrix[y1 + x] = (aliveCells == 3 || (aliveCells == 2 && s1->matrix[x + y1])) ? 1 : 0;
            }
        }
        swap((void **) &s1, (void **) &s2);
    }   
}