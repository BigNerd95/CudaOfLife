#include <openmp.h> 

void omp_compute_generation_singlefor(GenState_p s1, GenState_p s2){
    uint32_t dim = s1->rows * s1->cols;
    #pragma omp parallel for
    for (uint32_t i = 0; i < dim; i++) {

        uint32_t x1 = i % s1->cols;
        uint32_t x0 = (x1 - 1) % s1->cols;
        uint32_t x2 = (x1 + 1) % s1->cols;

        uint32_t y1 = i - x1;
        uint32_t y0 = (y1 - s1->cols) % dim;
        uint32_t y2 = (y1 + s1->cols) % dim;

        uint8_t aliveCells = countAliveCells(s1->matrix, x0, x1, x2, y0, y1, y2);
        s2->matrix[y1 + x1] = (aliveCells == 3 || (aliveCells == 2 && s1->matrix[y1 + x1])) ? 1 : 0;
        
    }
}

void omp_compute_generation(GenState_p s1, GenState_p s2){
    //#pragma omp parallel for
    for (uint32_t y = 0; y < s1->rows; y++) {
        uint32_t y0 = ((y - 1) % s1->rows) * s1->cols;
        uint32_t y1 = y * s1->cols;
        uint32_t y2 = ((y + 1) % s1->rows) * s1->cols;

        for (uint32_t x = 0; x < s1->cols; x++) {
            uint32_t x0 = (x - 1) % s1->cols;
            uint32_t x2 = (x + 1) % s1->cols;

            uint8_t aliveCells = countAliveCells(s1->matrix, x0, x, x2, y0, y1, y2);
            s2->matrix[y1 + x] = (aliveCells == 3 || (aliveCells == 2 && s1->matrix[x + y1])) ? 1 : 0;
        }
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void omp_compute_generation_pow2(GenState_p s1, GenState_p s2){
    uint32_t rows_m1 = s1->rows - 1;
    uint32_t cols_m1 = s1->cols - 1;

    //#pragma omp parallel for
    for (uint32_t y = 0; y < s1->rows; y++) {
        uint32_t y0 = ((y - 1) & rows_m1) * s1->cols;
        uint32_t y1 = y                   * s1->cols;
        uint32_t y2 = ((y + 1) & rows_m1) * s1->cols;

        for (uint32_t x = 0; x < s1->cols; x++) {
            uint32_t x0 = (x - 1) & cols_m1;
            uint32_t x2 = (x + 1) & cols_m1;

            uint8_t aliveCells = countAliveCells(s1->matrix, x0, x, x2, y0, y1, y2);
            s2->matrix[y1 + x] = (aliveCells == 3 || (aliveCells == 2 && s1->matrix[x + y1])) ? 1 : 0;
        }
    }
}

