#include <gui.h>

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

void simple(int rows, int cols, long int iterations){
    if (isPow2(rows) && isPow2(cols)){
        GenState_p s1 = create_gen(rows, cols);
        GenState_p s2 = create_gen(rows, cols);
        random_gen(s1);

        display_gen(s1);
        SDL_Delay(50);

        for (int i = 0; i < iterations; i++) {
            omp_compute_generation_pow2(s1, s2);
            swap((void **) &s1, (void **) &s2);

            display_gen(s1);
            SDL_Delay(50);
        }
    } else {
        puts("Rows or Cols are not a power of 2!");
    }
}

/*
int main(int argc, char *argv[]) {
    init_gui();
    srand((unsigned) time(0));

    simple(64, 64, 1000);

    quit_gui();
    return 0;
}*/