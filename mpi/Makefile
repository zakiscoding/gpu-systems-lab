# Roadmap step 5 (MPI second): point-to-point, collectives, and halo exchange.
#   make            build all binaries into build/
#   make run        run every experiment at its default rank count
#   make scaling    jacobi at 1, 2, 4, 8 ranks (strong scaling on one GPU)
#   make run-mps    same as run, under CUDA MPS (ranks share one GPU context;
#                   no root needed, daemon is stopped afterwards)
#
# nvcc compiles device code and hands host compilation/linking to mpicxx, so
# the MPI include and library flags come from Open MPI's wrapper.

NVCC      ?= /usr/local/cuda/bin/nvcc
CUDA_PATH ?= /usr/local/cuda
ARCH      ?= sm_120
NVCCFLAGS ?= -arch=$(ARCH) -O3 -lineinfo -std=c++17 -ccbin=mpicxx
MPIRUN    ?= mpirun
NP        ?= 4
BUILD := build

PROGS := pingpong allreduce jacobi
BINS  := $(addprefix $(BUILD)/,$(PROGS))

MPS_DIRS := CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log

.PHONY: all run run-mps scaling clean

all: $(BINS)

$(BUILD)/%: %.cu mpi_common.cuh ../cuda/common.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BUILD):
	mkdir -p $@

run: $(BINS)
	@set -e; \
	echo "=== pingpong (2 ranks) ==="; $(MPIRUN) -np 2 $(BUILD)/pingpong; echo; \
	echo "=== allreduce host ($(NP) ranks) ==="; $(MPIRUN) -np $(NP) $(BUILD)/allreduce; echo; \
	echo "=== allreduce GPU data ($(NP) ranks) ==="; $(MPIRUN) -np $(NP) $(BUILD)/allreduce --device; echo; \
	echo "=== jacobi ($(NP) ranks) ==="; $(MPIRUN) -np $(NP) $(BUILD)/jacobi

# Without MPS each rank has its own CUDA context and the GPU time-slices
# between them, which dominates small staged copies. MPS funnels all ranks
# through one server context so their work can run concurrently.
run-mps: $(BINS)
	@mkdir -p /tmp/mps-pipe /tmp/mps-log
	@export $(MPS_DIRS); nvidia-cuda-mps-control -d; \
	$(MAKE) --no-print-directory run MPIRUN="$(MPIRUN) -x CUDA_MPS_PIPE_DIRECTORY -x CUDA_MPS_LOG_DIRECTORY"; \
	status=$$?; echo quit | nvidia-cuda-mps-control; exit $$status

scaling: $(BUILD)/jacobi
	@set -e; for np in 1 2 4 8; do echo "=== jacobi np=$$np ==="; $(MPIRUN) -np $$np $(BUILD)/jacobi; echo; done

clean:
	rm -rf $(BUILD)
