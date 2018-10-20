#include <SDL2/SDL.h>
#include <openmp.h>
#include <sequential.h>  

#define WIDTH 640
#define HEIGHT 640
#define DELAY 50


//void color(int x, int y);
void clear_all();
void init_gui();
void quit_gui();
void display_gen(GenState_p gen);
void simple(uint32_t rows, uint32_t cols);