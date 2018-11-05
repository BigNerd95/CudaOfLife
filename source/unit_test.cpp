#include <unit_test.h>

void check_blinker(void (*compute_generation)(GenState_p s1, GenState_p s2, uint32_t iterations)){
    GenState_p blinker = create_gen(8, 8);
    GenState_p blinker2 = create_gen(8, 8);
    clear_gen(blinker);

    blinker->matrix[19] = 1;
    blinker->matrix[27] = 1;
    blinker->matrix[35] = 1;
    
    (*compute_generation)(blinker, blinker2, 1);
    
    for (int i = 0; i < 64; i++){
        if (i == 26 || i == 27 || i == 28) 
            assert(blinker2->matrix[i] == 1);
        else
            assert(blinker2->matrix[i] == 0);  
    }

    free_gen(blinker);
    free_gen(blinker2);
}

void check_beehive(void (*compute_generation)(GenState_p s1, GenState_p s2, uint32_t iterations)){
    GenState_p beehive = create_gen(8, 8);
    GenState_p beehive2 = create_gen(8, 8);

    clear_gen(beehive);
    beehive->matrix[19] = 1;
    beehive->matrix[20] = 1;
    beehive->matrix[26] = 1;
    beehive->matrix[29] = 1;
    beehive->matrix[35] = 1;
    beehive->matrix[36] = 1;
    
    (*compute_generation)(beehive, beehive2, 1);
    
    for (int i = 0; i < 64; i++){
        if (i == 19 || i == 20 || i == 26 || i == 29 || i == 35 || i == 36) 
            assert(beehive2->matrix[i] == 1);
        else
            assert(beehive2->matrix[i] == 0);  
    }

    free_gen(beehive);
    free_gen(beehive2);
}

void check_glinder(void (*compute_generation)(GenState_p s1, GenState_p s2, uint32_t iterations)){
    GenState_p glinder = create_gen(8, 8);
    GenState_p glinder2 = create_gen(8, 8);

    clear_gen(glinder);
    glinder->matrix[11] = 1;
    glinder->matrix[17] = 1;
    glinder->matrix[19] = 1;
    glinder->matrix[26] = 1;
    glinder->matrix[27] = 1;
    
    (*compute_generation)(glinder, glinder2, 1);
    for (int i = 0; i < 64; i++){
        if (i == 10 || i == 19 || i == 20 || i == 26 || i == 27) 
            assert(glinder2->matrix[i] == 1);
        else
            assert(glinder2->matrix[i] == 0);  
    }

    (*compute_generation)(glinder2, glinder, 1);
    for (int i = 0; i < 64; i++){
        if (i == 11 || i == 20 || i == 26 || i == 27 || i == 28) 
            assert(glinder->matrix[i] == 1);
        else
            assert(glinder->matrix[i] == 0);  
    }

    (*compute_generation)(glinder, glinder2, 1);
    
    for (int i = 0; i < 64; i++){
        if (i == 18 || i == 20 || i == 27 || i == 28 || i == 35) 
            assert(glinder2->matrix[i] == 1);
        else
            assert(glinder2->matrix[i] == 0);  
    }

    (*compute_generation)(glinder2, glinder,1);
   
    for (int i = 0; i < 64; i++){
        if (i == 20 || i == 26 || i == 28 || i == 35 || i == 36) 
            assert(glinder->matrix[i] == 1);
        else
            assert(glinder->matrix[i] == 0); 
    }
    
    free_gen(glinder);
    free_gen(glinder2);
}

void check_big_world(uint32_t rows, uint32_t cols, uint32_t iterations){
    GenState_p start = create_gen(rows, cols);
    GenState_p result_cpu = create_gen(rows, cols);
    GenState_p result_gpu = create_gen(rows, cols);
    random_gen(start);

    compute_cpu_generations_on_gpu_shared(start, result_gpu, iterations); //should run it first on gpu
    omp_compute_generations(start, result_cpu, iterations);
    
    assert(compare_gen(result_cpu, result_gpu));

    free_gen(start);
    free_gen(result_cpu);
    free_gen(result_gpu);
}

void check_multidimension(uint32_t rows, uint32_t cols, uint32_t iterations){
    GenState_p start = create_gen(rows, cols);
    GenState_p result_cpu = create_gen(rows, cols);
    GenState_p result_gpu = create_gen(rows, cols);
    random_gen(start);

    compute_cpu_generations_on_gpu_multidim(start, result_gpu, iterations); //should run it first on gpu
    omp_compute_generations(start, result_cpu, iterations);
    
    assert(compare_gen(result_cpu, result_gpu));

    free_gen(start);
    free_gen(result_cpu);
    free_gen(result_gpu);
}
    
int unit_testing_main(int argc, char *argv[]) {
    /*for (int i=0; i<argc; i++){
        printf("%s\n", argv[i]);
    }*/
    
    srand((unsigned) time(0));
    
    
    //Unit test sequential
    puts("Checking sequential functions");
    check_beehive((&compute_generations_singlefor));
    check_blinker((&compute_generations_singlefor));
    check_glinder((&compute_generations_singlefor));

    check_beehive(&compute_generations_doublefor);
    check_blinker(&compute_generations_doublefor);
    check_glinder(&compute_generations_doublefor);

    check_beehive(&seq_compute_generations);
    check_blinker(&seq_compute_generations);
    check_glinder(&seq_compute_generations);
    puts("All functions works correctly!\n");

    // Unit test OpenMp
    puts("Checking OpenMp functions");
    check_beehive(&omp_compute_generations);
    check_blinker(&omp_compute_generations);
    check_glinder(&omp_compute_generations);
    puts("All functions works correctly!\n");

    // Unit Test Gpu
    puts("Checking Cuda functions");
    check_beehive((&compute_cpu_generations_on_gpu));
    check_blinker((&compute_cpu_generations_on_gpu));
    check_glinder((&compute_cpu_generations_on_gpu));
    puts("All functions works correctly!\n");
    

    //Unit Test on Medium World
    puts("Testing Cuda on a medium world");
    check_big_world(64, 64, 1);
    check_big_world(64, 64, 2);
    check_big_world(64, 64, 3);
    puts("Test completed!\n");
    

    puts("Testing Cuda on a big world");
    check_big_world(128, 128, 2);
    check_big_world(1024, 1024, 2);
    puts("Test completed!\n");

    puts("Testing Cuda on a multidimension matrix");
    check_multidimension(MULTIDIM_R, MULTIDIM_C, 1); 
    puts("Test completed!\n");
    
    return 0;
}