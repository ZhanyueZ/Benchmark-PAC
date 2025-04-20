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

// since we target on UINT8 quantization
#define P_BITS 8
#define Q_BITS 8


// Pointer declaration
static uint8_t* X;
static uint8_t* Y;
static uint64_t* Y_host;
static uint64_t res = 0;

// Create input arrays
static void read_input(uint8_t* A, uint8_t* B, unsigned int nr_elements) {
    srand(0);
    printf("nr_elements\t%u\n", nr_elements);
    for (unsigned int i = 0; i < nr_elements; i++) {
        A[i] = (uint8_t) (rand() % 2);
        B[i] = (uint8_t) (rand() % 2);
    }
}

// Compute output in the host for verification purposes
/*
 * X : activation vector
 * W : weight vector
 * N : vector length
 * Thres : threshold for digital/Analog domain
*/

void pac_bitwise_dp(const uint8_t* X,
                        const uint8_t* W,
                        unsigned int N_exact,
                        unsigned int N_hybrid,
                        unsigned int Thres,
                        const uint32_t Sx_h[P_BITS],
                        const uint32_t Sw_h[Q_BITS],
                        uint64_t* out) 
{
    uint64_t res = 0;
    for(int i=0;i<N_exact;i++) {
        uint8_t x = X[i];
        uint8_t w = W[i];
        for(int p=0;p<P_BITS; p++) {
            uint8_t bit_x = (x >> p) & 1;
            if(!bit_x) continue;
            for(int q=0;q<Q_BITS;q++) {
                uint8_t bit_w = (w >> q) & 1;
                res += (bit_x & bit_w)  << (p + q);
            }
        }
    }

    for(unsigned int i=N_exact;i<N_exact + N_hybrid; i++) {
        uint8_t x = X[i];
        uint8_t w = W[i];
        for(int p = Thres;p<P_BITS;p++) {
            uint8_t bit_x = (x >> p) & 1;
            if(!bit_x) continue;
            for(int q = Thres; q< Q_BITS; q++) {
                if((w >> q) & 1) {
                    res += 1ULL << (p + q);
                }
            }
        }
    }

    uint64_t approx = 0;
    for(int p=0; p< P_BITS;p++) {
        for(int q=0; q<Q_BITS;q++) {
            if(!(p >= (int)Thres && q >= (int)Thres)) {
                uint64_t term = (uint64_t)Sx_h[p] * Sw_h[q] / N_hybrid;
                approx += term << (p+q);
            }
        }
    }

    res += approx;
    *out = res;
}



// Main of the Host Application
int main(int argc, char **argv) {

    // Input parameters
    struct Params p = input_params(argc, argv);

    // Timer declaration
    Timer timer;
#if defined(CYCLES) || defined(INSTRUCTIONS)
    double cc = 0;
    double cc_min = 0;
#endif
	

    // initialize the ratio

    double ratio = 0.1;
    

    // Allocate DPUs
    struct dpu_set_t dpu_set, dpu;
    uint32_t nr_of_dpus;
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus)); // Number of DPUs in the DPU set
    printf("Allocated %d DPU(s)\t", nr_of_dpus);
    printf("NR_TASKLETS\t%d\tBLOCK\t%d\n", NR_TASKLETS, BLOCK);

    // Load binary
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));

    // Input size 
    const unsigned int input_size = p.input_size; // Total input size 
    const unsigned int input_size_8bytes = 
        ((input_size * sizeof(uint8_t)) % 8) != 0 ? roundup(input_size, 8) : input_size; // Total input size, 8-byte aligned
    const unsigned int input_size_dpu = divceil(input_size, nr_of_dpus); // Input size per DPU (max.)
    const unsigned int input_size_dpu_8bytes = 
        ((input_size_dpu * sizeof(uint8_t)) % 8) != 0 ? roundup(input_size_dpu, 8) : input_size_dpu; // Input size per DPU (max.), 8-byte aligned

    

    // Input/output allocation in host main memory
    X = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(uint8_t));
    Y = malloc(input_size_dpu_8bytes * nr_of_dpus * sizeof(uint8_t));
    Y_host = malloc(sizeof(uint64_t));

    uint8_t *bufferX = X;
    uint8_t *bufferY = Y;
    
    unsigned int i = 0;

    // Create an input file with arbitrary data
    read_input(X, Y, input_size);

    uint32_t N_exact = (uint32_t)(input_size * ratio);
    uint32_t N_hybrid = input_size - N_exact;


    memset(Y_host, 0, sizeof(uint64_t));
    uint64_t *partial_res = aligned_alloc(8, nr_of_dpus*sizeof(uint64_t));
    memset(partial_res,0,nr_of_dpus * sizeof(uint64_t));

    uint32_t ele_per_dpu = input_size_dpu_8bytes;
    uint32_t exact_dpu_num = (N_exact + ele_per_dpu - 1) / ele_per_dpu;
    
    // collect only the hybrid ones 
    uint32_t Sx[P_BITS] = {0}, Sw[Q_BITS] = {0};
    for(unsigned i = N_exact; i < input_size; i++){
        uint8_t x = X[i], w = Y[i];
        for(int p=0; p<P_BITS; p++) Sx[p] += (x>>p)&1;
        for(int q=0; q<Q_BITS; q++) Sw[q] += (w>>q)&1;
    }

    // Loop over main kernel
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        // Compute output on CPU (verification purposes)
        if(rep >= p.n_warmup)
            start(&timer, 0, rep - p.n_warmup);
        pac_bitwise_dp(X, Y, N_exact,N_hybrid, 4, Sx, Sw, Y_host);  // we do 4 bit precision
        if(rep >= p.n_warmup)
            stop(&timer, 0);

        printf("Load input data\n");
        // Input arguments
        unsigned int kernel = 0;
        dpu_arguments_t input_arguments[NR_DPUS];
        for(i=0; i<nr_of_dpus-1; i++) {
            input_arguments[i].size=input_size_dpu_8bytes * sizeof(uint8_t); 
            input_arguments[i].transfer_size=input_size_dpu_8bytes * sizeof(uint8_t); 
            input_arguments[i].kernel=kernel;
            input_arguments[i].threshold = 4;
            input_arguments[i].total_elements = input_size;
            input_arguments[i].exact_count = N_exact;
            input_arguments[i].hybrid_count = N_hybrid;
            memcpy(input_arguments[i].Sx, Sx, sizeof(Sx));
            memcpy(input_arguments[i].Sw, Sw, sizeof(Sw));
            input_arguments[i].dpu_rank = i;
            input_arguments[i].num_exact_dpus = exact_dpu_num;
        }
        input_arguments[nr_of_dpus-1].size=(input_size_8bytes - input_size_dpu_8bytes * (NR_DPUS-1)) * sizeof(uint8_t); 
        input_arguments[nr_of_dpus-1].transfer_size=input_size_dpu_8bytes * sizeof(uint8_t); 
        input_arguments[nr_of_dpus-1].kernel=kernel;
        input_arguments[nr_of_dpus-1].threshold = 4;
        input_arguments[nr_of_dpus-1].total_elements = input_size;
        input_arguments[nr_of_dpus-1].exact_count = N_exact;
        input_arguments[nr_of_dpus-1].hybrid_count = N_hybrid;
        memcpy(input_arguments[nr_of_dpus-1].Sx, Sx, sizeof(Sx));
        memcpy(input_arguments[nr_of_dpus-1].Sw, Sw, sizeof(Sw));
        input_arguments[nr_of_dpus - 1].dpu_rank = nr_of_dpus-1;
        input_arguments[nr_of_dpus - 1].num_exact_dpus = exact_dpu_num;

        if(rep >= p.n_warmup)
            start(&timer, 1, rep - p.n_warmup); // Start timer (CPU-DPU transfers)
        i = 0;
        res = 0;
		// Copy input arguments
        // Parallel transfers
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments[i]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(input_arguments[0]), DPU_XFER_DEFAULT));

        // Copy input arrays
#ifdef SERIAL // Serial transfers

        //@@ INSERT SERIAL CPU-DPU TRANSFER HERE

#else // Parallel transfers

        //@@ INSERT PARALLEL CPU-DPU TRANSFER HERE
        // FIRST PUSH X
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, bufferX + input_size_dpu_8bytes * i));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set,DPU_XFER_TO_DPU,DPU_MRAM_HEAP_POINTER_NAME,0,input_size_dpu_8bytes * sizeof(uint8_t), DPU_XFER_DEFAULT));

        // then push y
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, bufferY + input_size_dpu_8bytes * i));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set,DPU_XFER_TO_DPU,DPU_MRAM_HEAP_POINTER_NAME,input_size_dpu_8bytes*sizeof(uint8_t), input_size_dpu_8bytes * sizeof(uint8_t), DPU_XFER_DEFAULT));



#endif
        if(rep >= p.n_warmup)
            stop(&timer, 1); // Stop timer (CPU-DPU transfers)
		
        printf("Run program on DPU(s) \n");
        // Run DPU kernel
        if(rep >= p.n_warmup) {
            start(&timer, 2, rep - p.n_warmup); // Start timer (DPU kernel)
        }
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
        if(rep >= p.n_warmup) {
            stop(&timer, 2); // Stop timer (DPU kernel)
        }

#if PRINT
        {
            unsigned int each_dpu = 0;
            printf("Display DPU Logs\n");
            DPU_FOREACH (dpu_set, dpu) {
                printf("DPU#%d:\n", each_dpu);
                DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
                each_dpu++;
            }
        }
#endif

        printf("Retrieve results\n");
        if(rep >= p.n_warmup)
            start(&timer, 3, rep - p.n_warmup); // Start timer (DPU-CPU transfers)
        i = 0;
        // Copy output array
#ifdef SERIAL // Serial transfers

        //@@ INSERT SERIAL DPU-CPU TRANSFER HERE

#else // Parallel transfers

        //@@ INSERT PARALLEL DPU-CPU TRANSFER HERE
        i = 0;
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, partial_res + i));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_size_dpu_8bytes * sizeof(uint8_t) + input_size_dpu_8bytes * sizeof(uint8_t), sizeof(uint64_t), DPU_XFER_DEFAULT));
        // final collect the res
        for(int i=0;i<nr_of_dpus;i++) {
            res += (uint64_t)partial_res[i];
        }

#endif
        if(rep >= p.n_warmup)
            stop(&timer, 3); // Stop timer (DPU-CPU transfers)

#if defined(CYCLES) || defined(INSTRUCTIONS)
        dpu_results_t results[nr_of_dpus];
        // Parallel transfers
        dpu_results_t* results_retrieve[nr_of_dpus];
        DPU_FOREACH(dpu_set, dpu, i) {
            results_retrieve[i] = (dpu_results_t*)malloc(NR_TASKLETS * sizeof(dpu_results_t));
            DPU_ASSERT(dpu_prepare_xfer(dpu, results_retrieve[i]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "DPU_RESULTS", 0, NR_TASKLETS * sizeof(dpu_results_t), DPU_XFER_DEFAULT));
        DPU_FOREACH(dpu_set, dpu, i) {
            results[i].count = 0;
            // Retrieve tasklet count
            for (unsigned int each_tasklet = 0; each_tasklet < NR_TASKLETS; each_tasklet++) {
                if (results_retrieve[i][each_tasklet].count > results[i].count)
                    results[i].count = results_retrieve[i][each_tasklet].count;
            }
            free(results_retrieve[i]);
        }

        uint64_t max_count = 0;
        uint64_t min_count = 0xFFFFFFFFFFFFFFFF;
        // Print performance results
        if(rep >= p.n_warmup){
            i = 0;
            DPU_FOREACH(dpu_set, dpu) {
                if(results[i].count > max_count)
                    max_count = results[i].count;
                if(results[i].count < min_count)
                    min_count = results[i].count;
                i++;
            }
            cc += (double)max_count;
            cc_min += (double)min_count;
        }
#endif
    }
#ifdef CYCLES
    printf("DPU cycles  = %g\n", cc / p.n_reps);
#elif INSTRUCTIONS
    printf("DPU instructions  = %g\n", cc / p.n_reps);
#endif
	
    // Print timing results
    printf("CPU ");
    print(&timer, 0, p.n_reps);
    printf("CPU-DPU ");
    print(&timer, 1, p.n_reps);
    printf("DPU Kernel ");
    print(&timer, 2, p.n_reps);
    printf("DPU-CPU ");
    print(&timer, 3, p.n_reps);

    // Check output
    bool status = true;

    if ((uint64_t)res != *Y_host) {
        status = false;
        printf("%llu(real value) -- %llu(dp returned from core) not matching", (unsigned long long)*Y_host, (unsigned long long)res);
    }

    else {
        printf("%llu -- %llu matched", (unsigned long long)*Y_host,(unsigned long long)res);
    }
    if (status) {
        printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    } else {
        printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
    }

    // Deallocation
    free(X);
    free(Y);
    free(Y_host);
    DPU_ASSERT(dpu_free(dpu_set)); // Deallocate DPUs
	
    return status ? 0 : -1;
}
