#include <gui.h>

SDL_Window *window;
SDL_Renderer *renderer;
int done;


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

    SDL_SetWindowResizable(window, SDL_FALSE);

    renderer = SDL_CreateSoftwareRenderer(SDL_GetWindowSurface(window));
    if(!renderer) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Render creation for surface fail : %s\n",SDL_GetError());
        exit(1);
    }

    done = 0;
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
    SDL_Rect rect = {0, 0, (int32_t) (darea.w/gen->cols), (int32_t) (darea.h/gen->rows)};
    
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);

    for (uint32_t r = 0; r < gen->rows; r++){
        for (uint32_t c = 0; c < gen->cols; c++){
            if (gen->matrix[r * gen->cols + c]){
                rect.y = r * rect.h; 
                rect.x = c * rect.w;
                SDL_RenderFillRect(renderer, &rect);
            }
        }
    }

    SDL_UpdateWindowSurface(window);

    SDL_Delay(DELAY);

    SDL_Event e;
    while(SDL_PollEvent(&e) != 0 ){
        if (e.type == SDL_QUIT){
            done = 1;
        }
    }
}

void run_gui(uint32_t rows, uint32_t cols, void (*compute_generation)(GenState_p s1, GenState_p s2, uint32_t iterations)){
    if (isPow2(rows) && isPow2(cols)){
        GenState_p s1 = create_gen(rows, cols);
        GenState_p s2 = create_gen(rows, cols);
        random_gen(s1);
        display_gen(s1);

        while (!done) {
            compute_generation(s1, s2, 1);
            swap((void **) &s1, (void **) &s2);

            display_gen(s1);
        }
        free_gen(s1);
        free_gen(s2);
    } else {
        puts("Rows or Cols are not a power of 2!");
    }
}


int gui_main(int argc, char *argv[]) {
    init_gui();
    srand((unsigned) time(0));

    if (argc > 1) {
        argc--;
        argv++;
        if (!strcmp("seq", *argv)){
            printf("on CPU\n");
            run_gui(128, 128, seq_compute_generations);

        } else if (!strcmp("omp", *argv)){
            printf("on OMP\n");
            run_gui(128, 128, omp_compute_generations);

        } else if (!strcmp("cuda", *argv)){
            printf("on CUDA\n");
            run_gui(128, 128, compute_cpu_generations_on_gpu);

        } else {
            printf("Unknown command!\n");
        }
    } else {
        run_gui(128, 128, seq_compute_generations);
    }

    quit_gui();
    return 0;
}