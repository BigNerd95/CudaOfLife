#ifndef __GOL_GUI__
#define __GOL_GUI__

#include <SDL2/SDL.h>
#include <sequential.h>  
#include <openmp.h>
#include <cuda.h>

#define WIDTH 640
#define HEIGHT 640
#define DELAY 50


//void color(int x, int y);
void clear_all();
void init_gui();
void quit_gui();
void display_gen(GenState_p gen);
void simple(uint32_t rows, uint32_t cols);

int gui_main(int argc, char *argv[]); 

#endif