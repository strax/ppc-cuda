NVCC ?= nvcc

NVCCFLAGS = \
	-std=c++17 \
	-Xcompiler "-march=native" \

so: so.cu
	$(NVCC) $(NVCCFLAGS) \
		-I .ppc/ppcso/include \
		-I .ppc/ppcgrader/include \
		-O3 \
		-lineinfo \
		.ppc/ppcso/tester.cc $^ -o $@
