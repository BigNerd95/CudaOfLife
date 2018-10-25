#include <SDL2/SDL.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 


/*
GUI: (https://wiki.libsdl.org/) (http://lazyfoo.net/tutorials/SDL)
- usa un solo rederer (non lo distrugge ogni volta)
- usa un solo rettangolo, lo riposiziona per colorare i pizel

Algo:
- dimensione righe multipla di 2 per poter usare and al posto del modulo

gcc sequenziale.c -o sequenziale -O2 -D_REENTRANT -lSDL2 && ./sequenziale
*/

#define WIDTH 640
#define HEIGHT 640
#define DELAY 50

typedef struct {
    unsigned char *matrix;
    unsigned int rows;
    unsigned int cols;
} GenState, *GenState_p;

SDL_Window *window;
SDL_Renderer *renderer;
int done;

void color(int x, int y){
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);

    SDL_Rect rect = { x, y, 100, 100};
    SDL_RenderFillRect(renderer, &rect);

    rect.x = x + 200;
    SDL_RenderFillRect(renderer, &rect);

    SDL_UpdateWindowSurface(window);
}

void clear_all(){
    SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, SDL_ALPHA_OPAQUE);
    SDL_RenderClear(renderer);
    SDL_UpdateWindowSurface(window);
}

void mia(){

    clear_all();
    SDL_Delay(2000);

    color(0,0);
    SDL_Delay(2000);

    clear_all();
    SDL_Delay(2000);

    color(100,100);
    SDL_Delay(2000);
}

void init_gui() {
    SDL_LogSetPriority(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_INFO);

    if(SDL_Init(SDL_INIT_VIDEO) != 0) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "SDL_Init fail : %s\n", SDL_GetError());
        exit(1);
    }

    window = SDL_CreateWindow("Game of Life", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WIDTH, HEIGHT, SDL_WINDOW_RESIZABLE);
    if(!window) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Window creation fail : %s\n",SDL_GetError());
        exit(1);
    }

    renderer = SDL_CreateSoftwareRenderer(SDL_GetWindowSurface(window));
    if(!renderer) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Render creation for surface fail : %s\n",SDL_GetError());
        exit(1);
    }
}

void quit_gui(){
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

void print_gen(GenState_p gen){
    printf("r: %d c: %d\n", gen->rows, gen->cols);

    for (int r = 0; r < gen->rows; r++){
        for (int c=0; c < gen->cols; c++){
            printf("%d ", gen->matrix[r * gen->cols + c]);
        }
        printf("\n");
    }
}

void display_gen(GenState_p gen){
    clear_all();

    SDL_Rect darea;
    SDL_RenderGetViewport(renderer, &darea);
    SDL_Rect rect = {0, 0, darea.w/gen->cols, darea.h/gen->rows};
    
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);

    for (int r = 0; r < gen->rows; r++){
        for (int c=0; c < gen->cols; c++){
            if (gen->matrix[r * gen->cols + c]){
                rect.y = r * rect.h; 
                rect.x = c * rect.w;
                SDL_RenderFillRect(renderer, &rect);
            }
        }
    }

    SDL_UpdateWindowSurface(window);
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

void clear_gen(GenState_p gen){
    memset(gen->matrix, 0, gen->cols * gen->rows);
}

void random_gen(GenState_p gen){
    for (int i=0; i < gen->cols * gen->rows; i++){
        gen->matrix[i] = rand() % 2;
    }
}

void simple(){
    GenState_p gen = create_gen(10, 10);
    for (int x=0; x<5; x++){
        random_gen(gen);
        //print_gen(gen);
        display_gen(gen);
        SDL_Delay(1000);
    }
}

void swap(void **a, void **b) {
    void *temp = *a;
    *a = *b;
    *b = temp;
}

unsigned char countAliveCells(unsigned char *matrix, size_t x0, size_t x1, size_t x2, size_t y0, size_t y1, size_t y2) {
    return matrix[x0 + y0] + matrix[x1 + y0] + matrix[x2 + y0] + matrix[x0 + y1]
         + matrix[x2 + y1] + matrix[x0 + y2] + matrix[x1 + y2] + matrix[x2 + y2];
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

void game(int rows, int cols){
    GenState_p s1 = create_gen(rows, cols);
    GenState_p s2 = create_gen(rows, cols);
    random_gen(s1);

    //display_gen(s1);
    //SDL_Delay(DELAY);

    for (int i = 0; i < 1000; i++) {

        compute_generation(s1, s2);

        swap((void **) &s1, (void **) &s2);
        //display_gen(s1);
        //SDL_Delay(DELAY);
    }
}

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

unsigned int log2pow2(unsigned int x){
    unsigned int pow2 = 0;
    while (x >>= 1){
        pow2++;
    }
    return pow2;
}

void compute_generation_pow2(GenState_p s1, GenState_p s2){
    unsigned int rows_m1 = s1->rows - 1;
    unsigned int cols_m1 = s1->cols - 1;
    //unsigned int cols_log2 = log2pow2(s1->cols);

    for (size_t y = 0; y < s1->rows; y++) {
        size_t y0 = ((y - 1) & rows_m1) * s1->cols;//<< cols_log2;
        size_t y1 = y                   * s1->cols;//<< cols_log2;
        size_t y2 = ((y + 1) & rows_m1) * s1->cols;//<< cols_log2;

        for (size_t x = 0; x < s1->cols; x++) {
            size_t x0 = (x - 1) & cols_m1;
            size_t x2 = (x + 1) & cols_m1;

            unsigned char aliveCells = countAliveCells(s1->matrix, x0, x, x2, y0, y1, y2);
            s2->matrix[y1 + x] = (aliveCells == 3 || (aliveCells == 2 && s1->matrix[x + y1])) ? 1 : 0;
        }
    }
}

int isPow2(unsigned int x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}

void game_pow2(int rows, int cols){
    if (isPow2(rows) && isPow2(cols)){
        GenState_p s1 = create_gen(rows, cols);
        GenState_p s2 = create_gen(rows, cols);
        random_gen(s1);

        for (int i = 0; i < 1000; i++) {

            compute_generation_pow2(s1, s2);

            swap((void **) &s1, (void **) &s2);
        }
    } else {
        puts("Rows or Cols are not a power of 2!");
    }
}

int main(int argc, char *argv[]) {
    //init_gui();

    srand((unsigned) time(0));

    //mia();
    //simple();
    //game(1024, 1024);
    game_pow2(1024, 1024);

    //SDL_Delay(5000);
    //getchar();
    //quit_gui();
    return 0;
}
