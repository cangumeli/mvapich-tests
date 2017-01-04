// System includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <cuda.h>
//#include <cuda_runtime.h>
// MPI include
#include <mpi.h>
// User include
#include "simpleMPI.h"

#define SEND_RECEIVE 0
#define REDUCE 1
#define BROADCAST 2

#define GPU_AWARE 1
#define GPU_UNAWARE 0

#define MATCH(s) (!strcmp(argv[i], (s)))
// Error handling macros
#define MPI_CHECK(call) \
    if((call) != MPI_SUCCESS) { \
        fprintf(stderr,"MPI error calling \""#call"\"\n");     \
        my_abort(-1); }

void send_receive_host(size_t buffer_size, int rank, double *time_elapsed, 
		       double *time_elapsed_with_alloc)
{
  float *buffer, *host;
  mallocGPU(&buffer, buffer_size);
  initDataGPU(buffer, buffer_size, 1);
  double t0 = MPI_Wtime(), t3;
  host = (float*)malloc(sizeof(float) * buffer_size);
  //if (rank == 0) initData(host, buffer_size, -1);
  /*if (rank == 0) {
    addScalarGPU(buffer, 1, 256, buffer_size/256);
    syncGPU();
  }*/
  MPI_Barrier(MPI_COMM_WORLD);
  //COMMUNICATION
  double t1, t2; //times
  MPI_Status status;
  t1 = MPI_Wtime();
  if (rank == 0) {
    copyGPU2HOST(host, buffer, buffer_size);
    MPI_Send(host, buffer_size, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    // printf("Host sum at 0: %f\n", sum(host, buffer_size)); 
  } else {
    MPI_Recv(host, buffer_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
    copyHOST2GPU(buffer, host, buffer_size);
  }
  t2 = MPI_Wtime();
  *time_elapsed = t2 - t1;
  /*if (rank == 1) {
    //Debug the transfer
    int num_amount;
    MPI_Get_count(&status, MPI_INT, &num_amount);
    printf("1 received %d numbers from 0\n", num_amount);
    printf("Sum of host data before receiver kernel launch %f\n", sum(host, buffer_size));
    addScalarGPU(buffer, 2, 256, buffer_size/256);
    syncGPU();
    copyGPU2HOST(host, buffer, buffer_size);
    printf("Sum of host data after receiver kernel launch %f\n", sum(host, buffer_size));
    printf("Total time %f\n", t2 - t1);
    printLastError();
    }*/
  free(host);
  t3 = MPI_Wtime();
  *time_elapsed_with_alloc = t3 - t0;
  freeGPU(buffer);
}

void send_receive_device(size_t buffer_size, int rank, double *time_elapsed)
{
  float *buffer, *host;
  mallocGPU(&buffer, buffer_size);
  initDataGPU(buffer, buffer_size, 1);
  printLastError();
  //host = (float*)malloc(sizeof(float) * buffer_size);
  //if (rank == 0) initData(host, buffer_size, -1);
  /*if (rank == 0) {
    addScalarGPU(buffer, 1, 256, buffer_size/256);
    syncGPU();
  }*/
  MPI_Barrier(MPI_COMM_WORLD);
  //COMMUNICATION
  double t1, t2; //times
  MPI_Status status;
  t1 = MPI_Wtime();
  if (rank == 0) {
    MPI_Send(buffer, buffer_size, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
  } else {
    MPI_Recv(buffer, buffer_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
  }
  t2 = MPI_Wtime();
  *time_elapsed = t2 - t1;
  /*if (rank == 1) {
    //Debug the transfer
    int num_amount;
    MPI_Get_count(&status, MPI_INT, &num_amount);
    //addScalarGPU(buffer, 2, 256, buffer_size/256);
    //syncGPU();
    //copyGPU2HOST(host, buffer, buffer_size);
    //printf("Sum of host data after receiver kernel launch %f\n", sum(host, buffer_size));
    //printf("Total time %f\n", t2 - t1);
    printLastError();
    }*/
  freeGPU(buffer);
  //free(host);
}


  
/*Test functions*/
// Host code
// No CUDA here, only MPI
int main(int argc, char* argv[]) {
    // Initialize MPI state
    MPI_CHECK(MPI_Init(&argc, &argv));
    int experiment=SEND_RECEIVE, gpu_aware=GPU_AWARE, i;
    for (i = 1; i < argc; ++i) {
      if (MATCH("-e")) {
	if (strcmp(argv[++i], "sr") == 0) {
	  experiment = SEND_RECEIVE;
	} else {
	  printf("Experiment is not defined yet\n");
	  MPI_Finalize();
	  return 1;
	}
      }
      
      else if (MATCH("-gpu")) {
	int a = atoi(argv[++i]);
	if (a) {
	  gpu_aware = GPU_AWARE;
	} else {
	  gpu_aware = GPU_UNAWARE; 
	}
      }
    }
    /*listDevices();*/
    
    // Get our MPI node number and node count
    int commSize, commRank;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &commSize));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &commRank));
    setDevice(commRank + 2);
    //send_receive_host(1024, commRank);
    size_t size;
    for (size = 2; size <= 4194304; size *= 2) {
      double time_elapsed = 0.0, alloc_time_elapsed;
      double temp = 0.0, temp2 = 0.0;
      if (gpu_aware == GPU_AWARE) {
	switch(experiment) {
	case SEND_RECEIVE:
	  for(int iter=0; iter<5; iter++){
	    send_receive_device(size, commRank, &time_elapsed);
	    temp += time_elapsed;
	  }
	  printf("size: %10d %c Average elapsed time: %f\n", size, '|', (temp/5));
	  break;
	default: break; 
	}
      } else { //GPU Unaware
	
	switch(experiment) {
	case SEND_RECEIVE:
	  for(int iter=0; iter<5; iter++){
	    send_receive_host(size, commRank, &time_elapsed, &alloc_time_elapsed);
	    temp += time_elapsed;
	    temp2 += alloc_time_elapsed;
	  }
	  printf("size: %10d %c Average elapsed time: %f General elapsed time %f\n", size, '|', (temp/5), (temp2/5));
	  break;
	default: break; 
	}
      }
    }
    // Generate some random numbers on the root node (node 0)
    MPI_CHECK(MPI_Finalize());
    if(commRank == 0) {
      printf("Benchmark is finalized\n");
    }
    return 0;
}

// Shut down MPI cleanly if something goes wrong
void my_abort(int err) {
    printf("MPI FAILED\n");
    MPI_Abort(MPI_COMM_WORLD, err);
}

