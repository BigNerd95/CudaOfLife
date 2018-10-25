#include <unit_test.h>
#include <gui.h>
#include <benchmark.h>


void usage(char *name){
    printf("Usage: %s (gui|test|bench) [seq|omp|cuda]\n", name);

}

int main(int argc, char **argv){
    if (argc > 1){
        argc--;
        argv++;
        if (!strcmp("gui", *argv)){
            printf("Running GUI\n");
            gui_main(argc, argv);

        } else if (!strcmp("test", *argv)){
            printf("Running Unit Testing\n");
            unit_testing_main(argc, argv);

        } else if (!strcmp("bench", *argv)){
            printf("Running Benchmark\n");
            benchmark_main(argc, argv);

        } else {
            printf("Unknown command!\n");
        }
    } else {
        usage(argv[0]);
    }
    return 0;
}