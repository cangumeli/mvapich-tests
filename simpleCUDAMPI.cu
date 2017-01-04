/* Simple example demonstrating how to use MPI with CUDA
*
*  Generate some random numbers on one node.
*  Dispatch them to all nodes.
*  Compute their square root on each node's GPU.
*  Compute the average of the results using MPI.
*
*  simpleMPI.cu: GPU part, compiled with nvcc
*/
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "simpleMPI.h"

// Error handling macro
#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        fprintf(stderr,"CUDA error calling \""#call"\", code is %d\n",err); \
        my_abort(err); }


// Device code
// Very simple GPU Kernel that computes square roots of input numbers
__global__ void simpleMPIKernel(float * buffer, float add) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    buffer[tid] += add;
}

void listDevices() {
  int nDevices;
  printf("here...");
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}

// Initialize an array with random data (between 0 and 1)
void initData(float * data, int dataSize, float scalar) {
  for(int i = 0; i < dataSize; i++) {
      data[i] = scalar;
  }
}

void initDataGPU(float *data, int dataSize, float initScalar)
{
  cudaMemset(data, initScalar, dataSize * sizeof(float));
}

void setDevice(int rank) {
  cudaError_t err = cudaSetDevice(rank);
  if (err != cudaSuccess) {
    printf("Error %s \n@ rank %d\n", cudaGetErrorString(err), rank);
  }
  /*else if (rank == 0) {
    double *dummy;
    cudaError_t errm = cudaMalloc(&dummy, 100000 * sizeof(float));
    if (errm != cudaSuccess)
      printf("Error %s \n@ rank %d\n", cudaGetErrorString(errm), rank);
    cudaDeviceSynchronize();
    }*/
}

void getInfo(int rank) {
  size_t freeBytes, totalBytes;
  cudaMemGetInfo(&freeBytes, &totalBytes);
  printf("Free bytes %lu, total bytes %lu, rank: %d\n", freeBytes, totalBytes, rank);
}

void mallocGPU(float **buffer, size_t size)  {
  cudaMalloc(buffer, size * sizeof(float));
}

void copyGPU2HOST(float *host_buffer, float *gpu_buffer, size_t size) {
  cudaMemcpy(host_buffer, gpu_buffer, size * sizeof(float), cudaMemcpyDeviceToHost);
}

void copyHOST2GPU(float *gpu_buffer, float *host_buffer, size_t size) {
  cudaMemcpy(gpu_buffer, host_buffer, size * sizeof(float), cudaMemcpyHostToDevice);
}

void addScalarGPU(float *gpu_buffer, float scalar, int blockSize, int gridSize) {
  dim3 threadsPerBlock(blockSize, 1, 1);
  dim3 blocksPerGrid(gridSize, 1, 1);
  simpleMPIKernel<<<threadsPerBlock, blocksPerGrid>>>(gpu_buffer, scalar);
}


void syncGPU() {
  cudaDeviceSynchronize();
}

void printLastError()
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Last error is %s\n", cudaGetErrorString(err));
  }
}


float sum(float * data, int size) {
  float accum = 0.f;
  for(int i = 0; i < size; i++) {
    accum += data[i];
    //printf("%d: %f\n", i, data[i]); 
  }
  return accum;
}

void freeGPU(float *buffer) {
  cudaFree(buffer);
}
