#ifndef _COMMON_H_
#define _COMMON_H_

// Transfer size between MRAM and WRAM
#ifdef BLOCK
#define BLOCK_SIZE_LOG2 BLOCK
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#else
#define BLOCK_SIZE_LOG2 8
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#define BLOCK BLOCK_SIZE_LOG2
#endif

// Data type
#ifdef INT32
#define T int32_t
#define DIV 2 // Shift right to divide by sizeof(T)
#elif INT64
#define T int64_t
#define DIV 3 // Shift right to divide by sizeof(T)
#elif FLOAT
#define T float
#define DIV 2 // Shift right to divide by sizeof(T)
#elif DOUBLE
#define T double
#define DIV 3 // Shift right to divide by sizeof(T)
#elif CHAR
#define T char
#define DIV 0 // Shift right to divide by sizeof(T)
#elif SHORT
#define T short
#define DIV 1 // Shift right to divide by sizeof(T)
#endif

// Structures used by both the host and the dpu to communicate information
typedef struct {
    uint32_t size;
    uint32_t transfer_size;
	enum kernels {
	    kernel1 = 0,
	    nr_kernels = 1,
	} kernel;
	uint32_t threshold;
	uint32_t total_elements;
	uint32_t Sx[8];
	uint32_t Sw[8];
	uint32_t dpu_rank;

	uint32_t exact_count;
	uint32_t hybrid_count;
	uint32_t num_exact_dpus;
} dpu_arguments_t; // Input arguments

typedef struct {
    uint64_t count;
} dpu_results_t; // Results (cycle count)

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define divceil(n, m) (((n)-1) / (m) + 1)
#define roundup(n, m) ((n / m) * m + m)

#endif
