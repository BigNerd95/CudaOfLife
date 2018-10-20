#include <sequential.h> 

void compute_generation_singlefor(GenState_p s1, GenState_p s2){
    size_t dim = s1->rows * s1->cols;
    for (size_t i = 0; i < dim; i++) {

        size_t x1 = i % s1->cols;
        size_t x0 = (x1 - 1) % s1->cols;
        size_t x2 = (x1 + 1) % s1->cols;

        size_t y1 = i - x1;
        size_t y0 = (y1 - s1->cols) % dim;
        size_t y2 = (y1 + s1->cols) % dim;

        unsigned char aliveCells = countAliveCells(s1->matrix, x0, x1, x2, y0, y1, y2);
        s2->matrix[y1 + x1] = (aliveCells == 3 || (aliveCells == 2 && s1->matrix[y1 + x1])) ? 1 : 0;
        
    }
}

void compute_generation(GenState_p s1, GenState_p s2){
    for (size_t y = 0; y < s1->rows; y++) {
        size_t y0 = ((y - 1) % s1->rows) * s1->cols;
        size_t y1 = y * s1->cols;
        size_t y2 = ((y + 1) % s1->rows) * s1->cols;

        for (size_t x = 0; x < s1->cols; x++) {
            size_t x0 = (x - 1) % s1->cols;
            size_t x2 = (x + 1) % s1->cols;

            unsigned char aliveCells = countAliveCells(s1->matrix, x0, x, x2, y0, y1, y2);
            s2->matrix[y1 + x] = (aliveCells == 3 || (aliveCells == 2 && s1->matrix[x + y1])) ? 1 : 0;
        }
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

void compute_generation_pow2(GenState_p s1, GenState_p s2){
    unsigned int rows_m1 = s1->rows - 1;
    unsigned int cols_m1 = s1->cols - 1;
    //unsigned int cols_log2 = log2pow2(s1->cols);

    for (size_t y = 0; y < s1->rows; y++) {
        size_t y0 = ((y - 1) & rows_m1) * s1->cols;
        size_t y1 = y                   * s1->cols;
        size_t y2 = ((y + 1) & rows_m1) * s1->cols;

        for (size_t x = 0; x < s1->cols; x++) {
            size_t x0 = (x - 1) & cols_m1;
            size_t x2 = (x + 1) & cols_m1;

            unsigned char aliveCells = countAliveCells(s1->matrix, x0, x, x2, y0, y1, y2);
            s2->matrix[y1 + x] = (aliveCells == 3 || (aliveCells == 2 && s1->matrix[x + y1])) ? 1 : 0;
        }
    }
}

