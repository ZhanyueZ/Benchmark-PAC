DPU_DIR := dpu
HOST_DIR := host
BUILDDIR ?= bin
NR_DPUS ?= 32
NR_TASKLETS ?= 16
BLOCK ?= 10
TYPE ?= CHAR
TRANSFER ?= PARALLEL
PRINT ?= 0
PERF ?= NO

define conf_filename
	${BUILDDIR}/.NR_DPUS_$(1)_NR_TASKLETS_$(2)_BLOCK_$(3)_TYPE_$(4)_TRANSFER_$(5)_PRINT_$(6)_PERF_$(7).conf
endef
CONF := $(call conf_filename,${NR_DPUS},${NR_TASKLETS},${BLOCK},${TYPE},${TRANSFER},${PRINT},${PERF})

HOST_TARGET := ${BUILDDIR}/host_code
DPU_TARGET := ${BUILDDIR}/dpu_code

COMMON_INCLUDES := support
HOST_SOURCES := $(wildcard ${HOST_DIR}/*.c)
DPU_SOURCES := $(wildcard ${DPU_DIR}/*.c)

.PHONY: all clean test

__dirs := $(shell mkdir -p ${BUILDDIR})

COMMON_FLAGS := -Wall -Wextra -g -I${COMMON_INCLUDES}
HOST_FLAGS := ${COMMON_FLAGS} -std=c11 -O3 `dpu-pkg-config --cflags --libs dpu` -DNR_TASKLETS=${NR_TASKLETS} -DNR_DPUS=${NR_DPUS} -DBLOCK=${BLOCK} -D${TYPE} -DPRINT=${PRINT} -D${TRANSFER} -D${PERF}
DPU_FLAGS := ${COMMON_FLAGS} -O2 -DNR_TASKLETS=${NR_TASKLETS} -DBLOCK=${BLOCK} -D${TYPE} -DPRINT=${PRINT} -D${TRANSFER} -D${PERF}

all: ${HOST_TARGET} ${DPU_TARGET}

${CONF}:
	$(RM) $(call conf_filename,*,*)
	touch ${CONF}

${HOST_TARGET}: ${HOST_SOURCES} ${COMMON_INCLUDES} ${CONF}
	$(CC) -o $@ ${HOST_SOURCES} ${HOST_FLAGS}

${DPU_TARGET}: ${DPU_SOURCES} ${COMMON_INCLUDES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -o $@ ${DPU_SOURCES}

clean:
	$(RM) -r $(BUILDDIR)

test: all
	./${HOST_TARGET}
