nvcc -c simpleCUDAMPI.cu

mpicxx -o simpleMPI simpleMPI.c simpleCUDAMPI.o -L/usr/local/cuda/lib64 -lcudart