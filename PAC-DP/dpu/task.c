/*
*  bitwise - dp with multiple tasklets
*
*/
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>

#include "../support/common.h"
#include "../support/cyclecount.h"


#define P_BITS 8
#define Q_BITS 8

// Input and output arguments
__host dpu_arguments_t DPU_INPUT_ARGUMENTS;
__host dpu_results_t DPU_RESULTS[NR_TASKLETS];

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

extern int main_kernel1(void);
int (*kernels[nr_kernels])(void) = {main_kernel1};
int main(void) { 
    // Kernel
    return kernels[DPU_INPUT_ARGUMENTS.kernel](); 
}



static uint32_t res_array[NR_TASKLETS];


// main_kernel1
int main_kernel1() {
    unsigned int tasklet_id = me();
#if PRINT
    printf("tasklet_id = %u\n", tasklet_id);
#endif
    if (tasklet_id == 0){ 
        mem_reset(); // Reset the heap
#ifdef CYCLES
        perfcounter_config(COUNT_CYCLES, true); // Initialize once the cycle counter
#elif INSTRUCTIONS
        perfcounter_config(COUNT_INSTRUCTIONS, true); // Initialize once the instruction counter
#endif
    }
    // Barrier
    barrier_wait(&my_barrier);
#if defined(CYCLES) || defined(INSTRUCTIONS)
    perfcounter_count count;
    dpu_results_t *result = &DPU_RESULTS[tasklet_id];
    result->count = 0;
    counter_start(&count); // START TIMER
#endif

    uint32_t input_size_dpu_bytes = DPU_INPUT_ARGUMENTS.size; // Input size per DPU in bytes
    uint32_t input_size_dpu_bytes_transfer = DPU_INPUT_ARGUMENTS.transfer_size; // Transfer input size per DPU in bytes
    uint32_t Thres = DPU_INPUT_ARGUMENTS.threshold;
    uint32_t N = DPU_INPUT_ARGUMENTS.total_elements;
    uint32_t *Sx = DPU_INPUT_ARGUMENTS.Sx;
    uint32_t *Sw = DPU_INPUT_ARGUMENTS.Sw;




    // Address of the current processing block in MRAM
    uint32_t base_tasklet = tasklet_id << BLOCK_SIZE_LOG2;
    uint32_t mram_base_addr_X = (uint32_t)DPU_MRAM_HEAP_POINTER;
    uint32_t mram_base_addr_Y = (uint32_t)(DPU_MRAM_HEAP_POINTER + input_size_dpu_bytes_transfer);
    uint32_t mram_base_addr_res = (uint32_t)(DPU_MRAM_HEAP_POINTER + 2*input_size_dpu_bytes_transfer);

    // Initialize a local cache in WRAM to store the MRAM block
	uint8_t *cache_X = (uint8_t *) mem_alloc(BLOCK_SIZE);
    uint8_t *cache_Y = (uint8_t *) mem_alloc(BLOCK_SIZE);
    uint32_t res = 0;

    for(unsigned int byte_index = base_tasklet; byte_index < input_size_dpu_bytes; byte_index += BLOCK_SIZE * NR_TASKLETS){
        // Bound checking
        //Since there are potentially a tasklet that operates on less than one data_block size
        uint32_t l_size_bytes = (byte_index + BLOCK_SIZE >= input_size_dpu_bytes) ? (input_size_dpu_bytes - byte_index) : BLOCK_SIZE;

        // Load cache with current MRAM block
        // MRAM-WRAM TRANSFERS 
        mram_read((__mram_ptr void const*)(mram_base_addr_X + byte_index), cache_X, l_size_bytes);
        mram_read((__mram_ptr void const*)(mram_base_addr_Y + byte_index), cache_Y, l_size_bytes);

        // for each tasklet - do the precise computing - all in parallel!
        for(uint32_t i=0; i<l_size_bytes;i++) {
            uint8_t a = cache_X[i], b = cache_Y[i];
            for (int p = Thres; p < P_BITS; p++) {
                if (!((a>>p)&1)) continue;
                for (int q = Thres; q < Q_BITS; q++) {
                    if ((b>>q)&1) {
                        res += 1 << (p + q);
                    }
                }
            }
        }
        
    }

    // for each tasklets hold it;
    res_array[tasklet_id] = res;
    // memory barrier to sync all tasklets
    barrier_wait(&my_barrier);
    // only one tasklet do the post-kernel write-back
    if(tasklet_id == 0) { 
        uint32_t exact = 0;
        for(int t=0;t<NR_TASKLETS;t++) exact += res_array[t];


        uint32_t rank = DPU_INPUT_ARGUMENTS.dpu_rank;
        uint64_t final = 0;
        if(rank == 0) {
            uint64_t approx = 0;
            for (int p = 0; p < P_BITS; p++) {
                for (int q = 0; q < Q_BITS; q++) {
                    if (!(p >= (int)Thres && q >= (int)Thres)) {
                        uint64_t term = (uint64_t)Sx[p] * Sw[q] / N;
                        approx += term << (p + q);
                    }
                }
            }
            final = (uint64_t)exact + approx;
        } else {
            final = exact;
        }
        
        mram_write(&final, (__mram_ptr void*)(mram_base_addr_res), sizeof(final));
    }

#if defined(CYCLES) || defined(INSTRUCTIONS)
    result->count += counter_stop(&count); // STOP TIMER
#endif
	
    return 0;
}
