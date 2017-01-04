//void computeGPU(float * hostData, int blockSize, int gridSize, float addAmount);
void my_abort(int err);
float sum(float * data, int size);

void initData(float * data, int dataSize, float initScalar);
void initDataGPU(float * data, int dataSize, float initScalar);
void listDevices();
void setDevice(int rank);
void getInfo(int rank);
void mallocGPU(float **buffer, size_t size);
void copyGPU2HOST(float *host_buffer, float *gpu_buffer, size_t size);
void copyHOST2GPU(float *gpu_buffer, float *host_buffer, size_t size);
void addScalarGPU(float *gpu_buffer, float scalar, int blockSize, int gridSize);
void syncGPU();
void printLastError();
void freeGPU(float *buffer);
