#include <sequential.h> 

void compute_generations_singlefor(GenState_p s1, GenState_p s2, uint32_t iterations){
    uint32_t dim = s1->rows * s1->cols;
    for (uint32_t iter = 0; iter < iterations; iter ++){
        
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
        swap((void **) &s1, (void **) &s2);
    }        
}

void compute_generations_doublefor(GenState_p s1, GenState_p s2, uint32_t iterations){
    for (uint32_t iter = 0; iter < iterations; iter ++){
        for (uint32_t y = 0; y < s1->rows; y++) {
            uint32_t y0 = ((y - 1) % s1->rows) * s1->cols;
            uint32_t y1 = y                    * s1->cols;
            uint32_t y2 = ((y + 1) % s1->rows) * s1->cols;

            for (uint32_t x = 0; x < s1->cols; x++) {
                uint32_t x0 = (x - 1) % s1->cols;
                uint32_t x2 = (x + 1) % s1->cols;

                uint8_t aliveCells = countAliveCells(s1->matrix, x0, x, x2, y0, y1, y2);
                s2->matrix[y1 + x] = (aliveCells == 3 || (aliveCells == 2 && s1->matrix[x + y1])) ? 1 : 0;
            }
        }
        swap((void **) &s1, (void **) &s2); 
    }
}

void seq_compute_generations(GenState_p s1, GenState_p s2, uint32_t iterations){
    uint32_t rows_m1 = s1->rows - 1;
    uint32_t cols_m1 = s1->cols - 1;
    uint32_t cols_p2 = log2pow2(s1->cols);

    for (uint32_t iter = 0; iter < iterations; iter ++){

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
