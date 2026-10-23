# Roadmap step 2: CUDA kernels + PTX / LLVM IR inspection.
#   make          build all binaries into build/
#   make run      build and run every benchmark (nonzero exit on mismatch)
#   make ptx      emit nvcc PTX for every kernel into build/ptx/
#   make llvm     emit clang CUDA device-side LLVM IR into build/llvm/

NVCC       ?= /usr/local/cuda/bin/nvcc
CUDA_PATH  ?= /usr/local/cuda
ARCH       ?= sm_120
NVCCFLAGS  ?= -arch=$(ARCH) -O3 -lineinfo -std=c++17

CLANG      ?= clang-21
CLANG_ARCH ?= $(ARCH)
BUILD := build

# clang 21 officially supports CUDA <= 12.8. With CUDA 13.x headers its
# runtime wrapper breaks in two places, patched here without touching the
# toolkit:
#   1. it #includes texture_fetch_functions.h, which CUDA 13 removed
#      -> an empty stand-in in $(COMPAT)
#   2. crt/math_functions.hpp uses _NV_RSQRT_SPECIFIER, which only the .h
#      (not included on clang's path) defines -> define it empty
# -Wno-unknown-cuda-version silences "CUDA version newer than supported".
# CUDA 13 moved CUB/Thrust/libcu++ to include/cccl (nvcc adds it, clang does not).
COMPAT     := $(BUILD)/clang-compat
CLANGFLAGS ?= -x cuda --cuda-gpu-arch=$(CLANG_ARCH) --cuda-path=$(CUDA_PATH) \
              --cuda-device-only -std=c++17 -Wno-unknown-cuda-version \
              -I$(COMPAT) -D_NV_RSQRT_SPECIFIER= -isystem $(CUDA_PATH)/include/cccl

PROGS := vector_add matmul reduction fused
BINS  := $(addprefix $(BUILD)/,$(PROGS))
LLVM_SRCS := vector_add matmul reduction fused

.PHONY: all run ptx llvm clean

all: $(BINS)

$(BUILD)/%: %.cu common.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(BUILD)/matmul: matmul.cu common.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $< -lcublas -Xlinker -rpath=$(CUDA_PATH)/lib64

$(BUILD):
	mkdir -p $@

run: $(BINS)
	@set -e; for p in $(PROGS); do echo "=== $$p ==="; ./$(BUILD)/$$p; echo; done

# PTX is the virtual ISA nvcc hands to ptxas; read it to see loads/stores,
# shuffles, shared-memory ops and unrolling. SASS: cuobjdump -sass build/<bin>.
ptx: $(addprefix $(BUILD)/ptx/,$(addsuffix .ptx,$(PROGS)))

$(BUILD)/ptx/%.ptx: %.cu common.cuh
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -ptx -o $@ $<

# Device-side LLVM IR from clang CUDA mode, at -O0 (straight from the
# frontend) and -O3 (after the optimizer), so the passes can be diffed.
llvm: $(foreach s,$(LLVM_SRCS),$(BUILD)/llvm/$(s).O0.ll $(BUILD)/llvm/$(s).O3.ll $(BUILD)/llvm/$(s).clang.ptx)

$(BUILD)/llvm/%.O0.ll: %.cu common.cuh $(COMPAT)/texture_fetch_functions.h
	@mkdir -p $(@D)
	$(CLANG) $(CLANGFLAGS) -O0 -Xclang -disable-O0-optnone -S -emit-llvm -o $@ $<

$(BUILD)/llvm/%.O3.ll: %.cu common.cuh $(COMPAT)/texture_fetch_functions.h
	@mkdir -p $(@D)
	$(CLANG) $(CLANGFLAGS) -O3 -S -emit-llvm -o $@ $<

# clang can also lower that IR to PTX through the LLVM NVPTX backend.
$(BUILD)/llvm/%.clang.ptx: %.cu common.cuh $(COMPAT)/texture_fetch_functions.h
	@mkdir -p $(@D)
	$(CLANG) $(CLANGFLAGS) -O3 -S -o $@ $<

clean:
	rm -rf $(BUILD)

$(COMPAT)/texture_fetch_functions.h:
	@mkdir -p $(@D)
	echo '// empty stand-in: removed in CUDA 13, still included by clang 21' > $@
