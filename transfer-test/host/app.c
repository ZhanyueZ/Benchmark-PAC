/**
* app.c
* Host Application Source File
*
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>

#include "../support/common.h"
#include "../support/timer.h"
#include "../support/params.h"

// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/dpu_code"
#endif

// Pointer declaration
static T* X;
static T* X_host;

// Create input arrays
static void read_input(T* A, unsigned int nr_elements) {
    //srand(0);
    printf("nr_elements\t%u\tnr_bytes\t%lu\n", nr_elements, nr_elements * sizeof(T));
    for (unsigned int i = 0; i < nr_elements; i++) {
        //A[i] = (T) (rand());
        A[i] = (T) 1;
    }
}

// Main of the Host Application
int main(int argc, char **argv) {

    // Input parameters
    struct Params p = input_params(argc, argv);

    // Timer declaration
    Timer timer;
	
    // Allocate DPUs
    struct dpu_set_t dpu_set, dpu;
    uint32_t nr_of_dpus;
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus)); // Number of DPUs in the DPU set
    printf("Allocated %d DPU(s)\t", nr_of_dpus);

    // Load binary
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));

    // Input size 
    const unsigned int input_size = p.input_size; // Total input size 
    const unsigned int input_size_dpu = divceil(input_size, nr_of_dpus); // Input size per DPU (max.)
    const unsigned int input_size_dpu_8bytes = 
        ((input_size_dpu * sizeof(T)) % 8) != 0 ? roundup(input_size_dpu, 8) : input_size_dpu; // Input size per DPU (max.), 8-byte aligned

    // Input/output allocation in host main memory
    X = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    X_host = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(T));
    T *bufferX = X;
    unsigned int i = 0;

    // Create an input file with arbitrary data
    read_input(X, input_size);
    memcpy(X_host, X, input_size_dpu_8bytes * nr_of_dpus * sizeof(T));

    // Loop over main kernel
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        printf("Load input data\n");
        if(rep >= p.n_warmup)
            start(&timer, 1, rep - p.n_warmup); // Start timer (CPU-DPU transfers)
        i = 0;
        // Copy input arrays
#ifdef SERIAL // Serial transfers

        //@@ INSERT SERIAL CPU-DPU TRANSFER HERE
        i = 0;
        DPU_FOREACH(dpu_set, dpu) {
            DPU_ASSERT(dpu_copy_to(dpu,DPU_MRAM_HEAP_POINTER_NAME, 0, bufferX + input_size_dpu_8bytes * i, input_size_dpu_8bytes * sizeof(T)));
            i++;
        }
        
#elif BROADCAST // Broadcast transfers

        //@@ INSERT BROADCAST CPU-DPU TRANSFER HERE
        DPU_ASSERT(dpu_broadcast_to(dpu_set,DPU_MRAM_HEAP_POINTER_NAME,0,bufferX,input_size_dpu * sizeof(T), DPU_XFER_DEFAULT));  // push the same portion of the buffer to each DPU

#else // Parallel transfers

        //@@ INSERT PARALLEL CPU-DPU TRANSFER HERE
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, bufferX + input_size_dpu_8bytes * i));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set,DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, input_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT));

#endif
        if(rep >= p.n_warmup)
            stop(&timer, 1); // Stop timer (CPU-DPU transfers)
		
        /*printf("Run program on DPU(s) \n");
        // Run DPU kernel
        if(rep >= p.n_warmup) {
            start(&timer, 2, rep - p.n_warmup); // Start timer (DPU kernel)
        }
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
        if(rep >= p.n_warmup) {
            stop(&timer, 2); // Stop timer (DPU kernel)
        }*/

        printf("Retrieve results\n");
        if(rep >= p.n_warmup)
            start(&timer, 3, rep - p.n_warmup); // Start timer (DPU-CPU transfers)
        i = 0;
        // Copy output array
#ifdef SERIAL // Serial transfers

        //@@ INSERT SERIAL DPU-CPU TRANSFER HERE
        i = 0;
        DPU_FOREACH(dpu_set, dpu) {
            DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, 0, X_host + input_size_dpu_8bytes * i, input_size_dpu_8bytes * sizeof(T)));
        }

#else // Parallel transfers

        //@@ INSERT PARALLEL DPU-CPU TRANSFER HERE
        DPU_FOREACH(dpu_set,dpu,i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, X_host + input_size_dpu * i));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, input_size_dpu * sizeof(T), DPU_XFER_DEFAULT));

#endif
        if(rep >= p.n_warmup)
            stop(&timer, 3); // Stop timer (DPU-CPU transfers)
    }
	
    // Print timing results
    printf("CPU-DPU ");
    print(&timer, 1, p.n_reps);
    printf("DPU Kernel ");
    print(&timer, 2, p.n_reps);
    printf("DPU-CPU ");
    print(&timer, 3, p.n_reps);

    // Check output
    bool status = true;
    for (i = 0; i < input_size; i++) {
        if(X_host[i] != X[i]){ 
            status = false;
            printf("%d: %lu -- %lu\n", i, X_host[i], X[i]);
        }
    }
    if (status) {
        printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    } else {
        printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
    }

    // Deallocation
    free(X);
    free(X_host);
    DPU_ASSERT(dpu_free(dpu_set)); // Deallocate DPUs
	
    return status ? 0 : -1;
}
