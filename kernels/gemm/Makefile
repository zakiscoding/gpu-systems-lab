# RTX 5070 = sm_120. CUTLASS (header-only) is optional: make CUTLASS=/path/to/cutlass
NVCC ?= nvcc
FLAGS = -O3 -std=c++17 -arch=sm_120 -lineinfo --expt-relaxed-constexpr
ifdef CUTLASS
FLAGS += -DWITH_CUTLASS -I$(CUTLASS)/include -I$(CUTLASS)/tools/util/include
endif
build/gemm: gemm.cu
	mkdir -p build && $(NVCC) $(FLAGS) -o $@ $< -lcublas
