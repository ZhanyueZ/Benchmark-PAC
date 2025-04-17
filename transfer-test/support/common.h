#ifndef _COMMON_H_
#define _COMMON_H_

// Data type
#define T int64_t
#define DIV 3 // Shift right to divide by sizeof(T)

// Structures used by both the host and the dpu to communicate information
typedef struct {
    uint32_t size;
    uint32_t transfer_size;
	enum kernels {
	    kernel1 = 0,
	    nr_kernels = 1,
	} kernel;
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
