// loop_paralelo.cu
#include <stdio.h>

__global__ void loop() {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%d\n", i);
}

int main() {
    loop<<<2, 5>>>();
    cudaDeviceSynchronize();
}