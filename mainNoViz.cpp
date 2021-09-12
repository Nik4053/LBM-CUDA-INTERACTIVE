// CUDA standard includes
#include "simMain.h"
#include <cuda_runtime.h>
#include <iostream>
#include "kernel.h"

cudaEvent_t start, stop;

inline void startCUDATimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
}

inline float printCUDATime(bool print) {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    if (print)
        std::cout << "Took: " << time << "ms" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return time;
}

int simMainNoViz(int argc, char **argv) {
    size_t frames = 600;
    std::cout << "Simulating " << frames << " frames..." << std::endl;
    int w = 1600;
    int h = 1600;
    uchar4 *imageCUDA;
    gpuErrchk(cudaMalloc((void **) &imageCUDA, w * h * sizeof(imageCUDA)));
    int2 loc = {0, 0};
    init(w,h);
    startCUDATimer();
    for (size_t i = 0; i < frames; ++i) {
        kernelLauncher(imageCUDA,loc);
    }
    //gpuErrchk( cudaPeekAtLastError() );
    float time = printCUDATime(false);
    std::cout << "Took: " << time << "ms.\t" << time / frames << "ms/frame.\t " << 1000 / time * frames << "fps"
              << std::endl;
    //kernel<<<grid,threads>>> ( d_odata, d_idata, size_x, size_y,
    //        NUM_REPS);

    return 0;
}