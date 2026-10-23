# Multi-machine MPI: RTX 5070 Ubuntu node + M1 iMac over Tailscale.
#
# Both machines use Open MPI 4.1.6 built from source with
# --enable-heterogeneous (x86_64 Linux and arm64 macOS disagree on
# sizeof(long double), and a stock build refuses to mix them).
#
#   make           build for this machine (CUDA on Linux, CPU on macOS)
#   make net       netbench across both machines   (run on the GPU node)
#   make jacobi    jacobi_hetero across both       (run on the GPU node)

OMPI      ?= $(HOME)/opt/openmpi-4.1.6-het
MPICXX    := $(OMPI)/bin/mpicxx
MPIRUN    := $(OMPI)/bin/mpirun
BUILD     := build
UNAME     := $(shell uname -s)

# Remote side (the iMac): its user, its install prefix, its repo checkout.
# The ssh user and key for MAC_HOST come from ~/.ssh/config on the GPU node.
MAC_HOST  ?= mos-imac
MAC_USER  ?= momo
MAC_OMPI  ?= /Users/$(MAC_USER)/opt/openmpi-4.1.6-het
MAC_DIR   ?= /Users/$(MAC_USER)/gpu-systems-lab/mpi/multinode
GPU_HOST  ?= $(shell hostname)
GPU_W     ?= 9
MAC_W     ?= 1
N         ?= 2048
STEPS     ?= 200

# Only TCP between machines; restrict to the Tailscale interface on Linux
# (macOS names it utunN, so let it pick there). orted on the Mac is found by
# absolute path since non-interactive ssh has no Open MPI on PATH.
MPIRUN_FLAGS := --mca btl tcp,self --mca oob_tcp_if_include 100.64.0.0/10 \
                --mca btl_tcp_if_include 100.64.0.0/10 \
                --mca orte_launch_agent $(MAC_OMPI)/bin/orted

# jacobi_hetero.cpp is plain C++ with an #ifdef USE_CUDA kernel: on Linux,
# nvcc compiles it as CUDA (-x cu) with mpicxx as host compiler; on macOS it
# is a CPU-only mpicxx build. netbench has no CUDA and uses mpicxx on both.
ifeq ($(UNAME),Linux)
NVCC  ?= /usr/local/cuda/bin/nvcc
# -fmad=false: no FMA contraction, so GPU results match the CPU bit for bit
JACOBI_BUILD = $(NVCC) -x cu -DUSE_CUDA -arch=sm_120 -O3 -fmad=false -std=c++17 -ccbin=$(MPICXX) -o $@ $<
else
# -ffp-contract=off: the CPU-side equivalent of -fmad=false (clang would
# otherwise contract a*b+c into FMA on arm64 and break the bit-exact check)
JACOBI_BUILD = $(MPICXX) -O3 -ffp-contract=off -std=c++17 -o $@ $<
endif

.PHONY: all net jacobi clean
all: $(BUILD)/netbench $(BUILD)/jacobi_hetero

$(BUILD)/netbench: netbench.cpp | $(BUILD)
	$(MPICXX) -O2 -std=c++17 -o $@ $<

$(BUILD)/jacobi_hetero: jacobi_hetero.cpp | $(BUILD)
	$(JACOBI_BUILD)

$(BUILD):
	mkdir -p $@

# MPMD launch: `a : b` starts one rank per host, each with its own binary path
# (the Mac's checkout lives elsewhere) and, for jacobi, its own --weight.
# Both machines must have built `make` first.
net: $(BUILD)/netbench
	$(MPIRUN) $(MPIRUN_FLAGS) \
	  -np 1 --host $(GPU_HOST) $(CURDIR)/$(BUILD)/netbench : \
	  -np 1 --host $(MAC_HOST) $(MAC_DIR)/$(BUILD)/netbench

jacobi: $(BUILD)/jacobi_hetero
	$(MPIRUN) $(MPIRUN_FLAGS) \
	  -np 1 --host $(GPU_HOST) $(CURDIR)/$(BUILD)/jacobi_hetero --n $(N) --steps $(STEPS) --weight $(GPU_W) : \
	  -np 1 --host $(MAC_HOST) $(MAC_DIR)/$(BUILD)/jacobi_hetero --n $(N) --steps $(STEPS) --weight $(MAC_W)

clean:
	rm -rf $(BUILD)
