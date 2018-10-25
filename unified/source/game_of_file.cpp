#include <unit_test.h>
#include <gui.h>
#include <cudax.h>

void usage(){
    printf("Usage:\n");

}

int main(int argc, char **argv){
    if (argc > 1){
        if (!strcmp("gui", argv[1])){
            printf("Avvio GUI\n");
            gui_main();

        } else if (!strcmp("test", argv[1])){
            printf("Avvio Unit Testing\n");
            unit_testing_main();

        } else if (!strcmp("bench", argv[1])){
            printf("Avvio Benchmark\n");
            benchmark_main();

        } else {
            usage();
        }
    } else {
        usage();
    }
    return 0;
}