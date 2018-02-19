# Build tools
CFLAGS= -I/usr/local/cuda-9.1/samples/common/inc
NVCC = /usr/local/cuda-9.1/bin/nvcc


# here are all the objects
GPUOBJS = main.o colour-convert.o
OBJS = cuda.o

# make and compile
mycode.out: $(GPUOBJS)
	$(NVCC) -o mycode $(GPUOBJS) 

main.o: main.cpp
	$(NVCC) -c main.cpp $(CFLAGS)

colour-convert.o: colour-convert.cu
	$(NVCC) -c colour-convert.cu $(CFLAGS)

clean:
	rm out_rgb.ppm out_yuv.yuv $(GPUOBJS)

