#include "kernel.h"
#include "stdio.h"
#include "math.h"
#include "device_launch_parameters.h"
#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif
#include <iostream>
#include <cstdlib>
#define PRECISION double

#define TX 32
#define TY 32
//#define DEBUG

// 3d version
const dim3 blockSize(TX, TY, 1);
dim3 gridSize;

struct GpuData* gpuDataCPU; // used for freeing all cuda memory again
__device__
struct GpuData* gpuDataLOCAL; // do not use in device code!!!


enum DIR {
    C = 0,
    W = 1,
    E = 2,
    N = 3,
    S = 4,
    NW = 5,
    NE = 6,
    SW = 7,
    SE = 8
};

enum CellType
{
    FLUID_CELL = 0,
    NO_SLIP_BOUNDARY = 1,
    VELOCITY_BOUNDARY = 2,
    DENSITY_BOUNDARY = 3,
    OBSTACLE_BOUNDARY = 4
};



struct GpuData {
    int w;
    int h;
    CellType* type;
    PRECISION* velY, * velX, * density, * forceY, * forceX;
    PRECISION* grid[9];
    PRECISION* grid_tmp[9];
    PRECISION relaxTime, velInX, velInY,Re;
    int cIdx[9];
};

// lattice velocities
__device__
int cx[9] = { 0, -1, 1, 0, 0, -1, 1, -1, 1 }; 
int cxCPU[9] = { 0, -1, 1, 0, 0, -1, 1, -1, 1 };
__device__
int cy[9] = { 0, 0, 0, -1, 1, -1, -1, 1, 1 };
int cyCPU[9] = { 0, 0, 0, -1, 1, -1, -1, 1, 1 };
//__device__
//int cIdx[9]; // contains the number that has to be add/sub from the idx to get new postion for cx,cy
__device__
DIR DirInverse[9] = { C,E,W,S,N,SE,SW,NE,NW };
// equilibrium weights
__device__
PRECISION weights[9] = { 4.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9,
1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36 };
PRECISION weightsCPU[9] = { 4.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9,
1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36 };

__device__
unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

__device__
void fillDensity(GpuData* gd, const int idx) {
    PRECISION tmp = 0;
    for (size_t i = 0; i < 9; i++)
    {
        tmp += gd->grid[i][idx];
    }
    gd->density[idx] = tmp;
}
__device__
void fillVelocity(GpuData* gd, const int idx) {
    gd->velY[idx] = 0;
    gd->velX[idx] = 0;
    for (size_t i = 0; i < 9; i++)
    {
        gd->velY[idx] += gd->grid[i][idx] * cy[i];
        gd->velX[idx] += gd->grid[i][idx] * cx[i];
    }
}
__device__
void collideFluidCells(GpuData* gd, const int y, const int x) {
    const int idx = x + y * gd->w; // 1D indexing
    if (gd->type[idx] != FLUID_CELL)return;
    fillDensity(gd, idx);
    fillVelocity(gd, idx);
    PRECISION density = gd->density[idx];
    PRECISION velX = gd->velX[idx], velY = gd->velY[idx];

    // collide:
    for (size_t i = 0; i < 9; i++)
    {
        double uTu = velX * velX + velY * velY;
        double cTu = cy[i] * velY + cx[i] * velX;
        double feq = weights[i] * (density  + 3 * cTu + 9.0 / 2 * cTu * cTu - 3.0 * uTu/2.0);
        gd->grid[i][idx] = gd->grid[i][idx] - (1.0 / gd->relaxTime) * (gd->grid[i][idx] - feq);
    }
    
#ifdef DEBUG
    fillDensity(gd, idx);
    PRECISION den2 = gd->density[idx];
    if (den2 > density + 0.01 || den2 < density - 0.01) {
        printf("Fluid density changed from %f to %f \n", density, den2);
    }
    fillVelocity(gd, idx);
    PRECISION velX2 = gd->velX[idx], velY2 = gd->velY[idx];
    if (velX2 > velX + 0.01 || velX2 < velX - 0.01) {
        printf("Fluid velX changed from %f to %f \n", velX, velX2);
    }
    if (velY2 > velY + 0.01 || velY2 < velY - 0.01) {
        printf("Fluid velX changed from %f to %f \n", velY, velY2);
    }
#endif
}




__device__
void preprocessBoundaries(GpuData* gd, const int y, const int x) {
    const int idx = x + y * gd->w; // 1D indexing
    // do nothing for fluid cells
    if (gd->type[idx] == FLUID_CELL)return;
    // reflection for no_slip, obstacle and velocity boundary
    if (gd->type[idx] == DENSITY_BOUNDARY)return;

    if (gd->type[idx] == OBSTACLE_BOUNDARY)
    {
        gd->forceX[idx] = 0;
        gd->forceY[idx] = 0;
    }

    int maxIdx = gd->h * gd->w;
    // pull only from FLUID
    for (size_t i = 0; i < 9; i++)
    {
        // if (y == 0 && cy[i] == -1) continue;
        // if (x == 0 && cx[i] == -1) continue;
        // if (y == gd->h - 1 && cy[i] == 1) continue;
        // if (x == gd->w - 1 && cx[i] == 1) continue;
        int dstIdx = idx + gd->cIdx[i];//idx + cy[i] * gd->w + cx[i];// 
        if (y == 0 && cy[i] == -1) continue;
        if (cy[i] + y >= gd->h) continue;
        if (x == 0 && cx[i] == -1) continue;
        if (cx[i] + x >= gd->w) continue;

        //if ( cx[i] + x == gd->w || cx[i] + x == -1 || cy[i] + y == gd->h || cy[i] + y == -1) continue; // dont wrap around handling
        if (dstIdx < 0 || dstIdx >= maxIdx) continue; //check bounds TODO can be removed
        if (gd->type[dstIdx] != FLUID_CELL) continue;
        gd->grid[i][idx] = gd->grid[DirInverse[i]][dstIdx]; // pull 
        if (gd->type[idx] == OBSTACLE_BOUNDARY) // calculate forces
        {
            gd->forceX[idx] += 2 * gd->grid[i][idx] * cx[i];
            gd->forceY[idx] += 2 * gd->grid[i][idx] * cy[i];
        }

    }
}

__device__
void handleBoundaries(GpuData* gd, const int y, const int x) {
    const int idx = x + y * gd->w; // 1D indexing
    // do nothing for fluid cells
    if (gd->type[idx] == FLUID_CELL)return;

    // boundary handling velocity
    else if (gd->type[idx] == VELOCITY_BOUNDARY)
    {
        for (size_t i = 0; i < 9; i++)
        {
            double cu = cx[i] * gd->velInX + cy[i] * gd->velInY;
            gd->grid[i][idx] = gd->grid[i][idx] + /*+*/ 6 * weights[i] * cu; // TODO here ERROR
        }
    }
    //boundary handling density
    else if (gd->type[idx] == DENSITY_BOUNDARY)
    {
        fillVelocity(gd, idx);
        double velX= gd->velX[idx], velY= gd->velY[idx];
        for (size_t i = 0; i < 9; i++)
        {
            double cu = cx[i] * velX + cy[i] * velY;
            double rho_out = 1; // TODO param
            double uu = velX * velX + velY * velY;
            gd->grid[i][idx] = - gd->grid[i][idx] + 2 * weights[i] * (rho_out + (9.0 / 2.0) * cu * cu - (3.0 / 2.0) * uu);
        }
    }
}


__device__
void stream(GpuData* gd, const int y, const int x) {
    const int idx = x + y * gd->w; // 1D indexing
   // only for fluid cells
    if (gd->type[idx] != FLUID_CELL)return;
    // pull
    for (size_t i = 0; i < 9; i++)
    {
        gd->grid_tmp[i][idx] = gd->grid[i][idx +gd->cIdx[DirInverse[i]]];
    }
}
__global__
void step1(GpuData* gd) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= gd->w) || (y >= gd->h)) return; // Check if within image bounds
    collideFluidCells(gd, y, x);
}
__global__
void step2(GpuData* gd) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= gd->w) || (y >= gd->h)) return; // Check if within image bounds
    preprocessBoundaries(gd, y, x);
    handleBoundaries(gd, y, x);
}

__global__
void step3(GpuData* gd) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= gd->w) || (y >= gd->h)) return; // Check if within image bounds
    handleBoundaries(gd, y, x);
}



__device__
void swapData(GpuData* gd, const int y, const int x) {
    const int idx = x + y * gd->w; // 1D indexing
    /*PRECISION* tmp[9];
    for (size_t i = 0; i < 9; i++)
    {
        tmp[i] = gd->grid[i];
    }
    for (size_t i = 0; i < 9; i++)
    {
        gd->grid[i] = gd->grid_tmp[i];
    }
    for (size_t i = 0; i < 9; i++)
    {
        gd->grid_tmp[i] = tmp[i];
    }*/
    for (size_t i = 0; i < 9; i++)
    {
        gd->grid[i][idx] = gd->grid_tmp[i][idx];
    }
}

__global__
void step4(GpuData* gd) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= gd->w) || (y >= gd->h)) return; // Check if within image bounds
    stream(gd, y, x);
    //swapData(gd, y, x);
}
__global__
void step5(GpuData* gd) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= gd->w) || (y >= gd->h)) return; // Check if within image bounds
    swapData(gd, y, x);
}

__global__
void imageKernelDensity(uchar4* d_out, GpuData* gd) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= gd->w) || (y >= gd->h)) return; // Check if within image bounds
    const int idx = x + y * gd->w; // 1D indexing
    fillDensity(gd, idx);
    fillVelocity(gd, idx);
    d_out[idx].x = clip(fabs(gd->density[idx] - 0.8) * 300);
    d_out[idx].y = 0;// clip(fabs(gd->velY[idx] * 10000));
    d_out[idx].z = 0;
    d_out[idx].w = 255;

}

__global__
void imageKernelType(uchar4* d_out, GpuData* gd) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= gd->w) || (y >= gd->h)) return; // Check if within image bounds
    const int idx = x + y * gd->w; // 1D indexing
    d_out[idx].x = 0;
    d_out[idx].y = 0;
    d_out[idx].z = 0;
    switch (gd->type[idx])
    {
    case FLUID_CELL:
        //d_out[idx].x = 255;
        break;
    case NO_SLIP_BOUNDARY:
        d_out[idx].x = 255;
        d_out[idx].y = 255;
        break;
    case VELOCITY_BOUNDARY:
        d_out[idx].z = 100;
        break;
    case DENSITY_BOUNDARY:
        d_out[idx].z = 255;
        break;
    case OBSTACLE_BOUNDARY:
        d_out[idx].y = 255;
        break;
    default:
        break;
    }
    d_out[idx].w = 255;
}
__global__
void imageKernel(uchar4* d_out, GpuData* gd) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= gd->w) || (y >= gd->h)) return; // Check if within image bounds
    const int idx = x + y * gd->w; // 1D indexing
    fillDensity(gd, idx);
    fillVelocity(gd, idx);
    d_out[idx].x = clip(fabs(gd->velX[idx] * 10000));
    d_out[idx].y = clip(fabs(gd->velY[idx] * 10000));
    d_out[idx].z = 0;
    d_out[idx].w = 255;

}
__global__
void imageKernelMagnitude(uchar4* d_out, GpuData* gd) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= gd->w) || (y >= gd->h)) return; // Check if within image bounds
    const int idx = x + y * gd->w; // 1D indexing
    fillDensity(gd, idx);
    fillVelocity(gd, idx);
    double vel = gd->velX[idx] * gd->velX[idx] + gd->velY[idx] * gd->velY[idx];
    d_out[idx].x = clip(fabs(sqrt(vel) * 5000)); //clip(fabs(vel * 100000));//
    d_out[idx].y = 0;// clip(fabs(gd->velY[idx] * 10000));
    d_out[idx].z = 0;
    d_out[idx].w = 255;// clip(fabs(vel * 100000)); ;

}
__global__
void fillDensityVelocityKernel(GpuData* gd) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= gd->w) || (y >= gd->h)) return; // Check if within image bounds
    const int idx = x + y * gd->w; // 1D indexing
    fillDensity(gd, idx);
    fillVelocity(gd, idx);
}
int ww, hh;
void kernelLauncher(uchar4 *d_out, int2 pos) {
#ifdef DEBUG
    fillDensityVelocityKernel << <gridSize, blockSize >> > (gpuDataLOCAL);
    cudaMemcpy(gpuDataCPU, gpuDataLOCAL,  sizeof(GpuData), cudaMemcpyDeviceToHost); gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize(); gpuErrchk(cudaPeekAtLastError());
    PRECISION* velXCPU = (PRECISION*)malloc(ww * hh * sizeof(PRECISION));
    cudaMemcpy(velXCPU, gpuDataCPU->velX, ww * hh * sizeof(PRECISION), cudaMemcpyDeviceToHost); gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize(); gpuErrchk(cudaPeekAtLastError());
    printf("Printing VEl:\n");
    for (size_t y = 0; y < hh; y++)
    {
        for (size_t x = 0; x < ww; x++)
        {
            printf("%f\t", velXCPU[y * ww + x]);
        }
        printf("\n");
    }
    free(velXCPU);
#endif // DEBUG
    //Sleep(20);
    for (size_t iter = 0; iter < 100; iter++)
    {
        step1 << <gridSize, blockSize >> > (gpuDataLOCAL);
        gpuErrchk(cudaPeekAtLastError());
        cudaDeviceSynchronize(); gpuErrchk(cudaPeekAtLastError());
        step2 << <gridSize, blockSize >> > (gpuDataLOCAL);
        gpuErrchk(cudaPeekAtLastError());
        cudaDeviceSynchronize(); gpuErrchk(cudaPeekAtLastError());

        //step3 << <gridSize, blockSize >> > (gpuDataLOCAL);
        //gpuErrchk(cudaPeekAtLastError());
        //cudaDeviceSynchronize(); gpuErrchk(cudaPeekAtLastError());
        step4 << <gridSize, blockSize >> > (gpuDataLOCAL);
        gpuErrchk(cudaPeekAtLastError());
        cudaDeviceSynchronize(); gpuErrchk(cudaPeekAtLastError());
        step5 << <gridSize, blockSize >> > (gpuDataLOCAL);
        gpuErrchk(cudaPeekAtLastError());
        cudaDeviceSynchronize(); gpuErrchk(cudaPeekAtLastError());

    }

    imageKernelMagnitude << <gridSize, blockSize >> > (d_out, gpuDataLOCAL);
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize(); gpuErrchk(cudaPeekAtLastError());
   
}


// setting tags and non_sense_weights for given sphere obstacle
void setObstacleSphere(CellType* type, PRECISION** grid, size_t spherey, size_t spherex, size_t diameter, size_t sizey, size_t sizex, PRECISION nonsense_weights[9])
{
    // going through all points wich are in a square(with center point (spherex, spherey)) with length diameter
    for (int y = (int)spherey - (int)diameter; y < (int)spherey + (int)diameter - 1; y++)
    {
        for (int x = (int)spherex - (int)diameter; x < (int)spherex + (int)diameter - 1; x++)
        {
            // outside the grid
            if (x < 0 || y < 0 || x >= (int)sizex || y >= (int)sizey)
            {
                continue;
            }
            // choosing only the cells with center point in side the given circle
            double dist = sqrt(pow((double)x + 0.5 - (double)spherex, 2.0) + pow((double)y + 0.5 - (double)spherey, 2.0));
            if (dist < (double)diameter / 2.0)
            {
                type[x + y * (int)sizex] = OBSTACLE_BOUNDARY;
                for (size_t key = 0; key < 9; key++)
                {
                    grid[key][x + y * (int)sizex] = nonsense_weights[key];
                }
            }
        }
    }
}

void initgrid(CellType* type, PRECISION** grid,int width, int height) {
    PRECISION nonsense_weights[9] = { .7 / 9,.3 / 9,.6 / 9,.6 / 9,.5 / 9,.4 / 36,.4 / 36,.2 / 36,.7 / 36 };
    //PRECISION nonsense_weights[9] = { 4.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9,1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36 };
    for (size_t y = 0; y < height; y++)
    {
        for (size_t x = 0; x < width; x++)
        {
            int idx = y * width + x;
            if (y == 0 || y == height - 1) {
                type[idx] = NO_SLIP_BOUNDARY;
                for (size_t key = 0; key < 9; key++)
                {
                    grid[key][idx] =  nonsense_weights[key];
                }
            }
            else if(x==width-1)
            {
                type[idx] = DENSITY_BOUNDARY;
                for (size_t key = 0; key < 9; key++)
                {
                    grid[key][idx] = weightsCPU[key];
                }
            }
            else if (x == 0)
            {
                type[idx] = VELOCITY_BOUNDARY;
                for (size_t key = 0; key < 9; key++)
                {
                    grid[key][idx] = nonsense_weights[key];
                }
            }
            else
            {
                type[idx] = FLUID_CELL;
                for (size_t key = 0; key < 9; key++)
                {
                    grid[key][idx] = weightsCPU[key];
                }
            }
            //type[idx] = NO_SLIP_BOUNDARY;
        }
    }
    //setObstacleSphere(type, grid, 60 , 200, 20, height, width, nonsense_weights);
    setObstacleSphere(type, grid, 40*1, 100*1, 20*1, height, width, nonsense_weights);
}


void init(int w, int h) {
    ww = w;
    hh = h;
    gridSize = dim3((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y,
        1); // + TX - 1 for w size that is not divisible by TX
    // alloc cpu version of struct
    //GpuData tmp = { w,h };
    gpuDataCPU = (GpuData*)malloc(sizeof(GpuData));
    gpuDataCPU->w = w;
    gpuDataCPU->h = h;
    gpuDataCPU->velInX = 0.02;
    gpuDataCPU->velInY = 0; // Not supported
    gpuDataCPU->Re = 4000;
    gpuDataCPU->relaxTime = 3 * gpuDataCPU->velInX * ((double)h - 2) / gpuDataCPU->Re + 0.5;
    size_t arrSize = gpuDataCPU->h * gpuDataCPU->w;
    CellType* typeCPU = (CellType*)malloc(arrSize * sizeof(CellType));
    PRECISION* gridCPU[9];// = { 0,0,0,0,0,0,0,0,0 }; //(PRECISION**)malloc(9 * sizeof(PRECISION*));
    for (size_t i = 0; i < 9; i++)
    {
        gridCPU[i] = (PRECISION*)malloc(arrSize * sizeof(PRECISION));
        //if (gridCPU[i] == NULL) {
        //    printf("Error allocating.\n");
       // }
    }
    //PRECISION* gridCPU2 = new PRECISION(arrSize);
    initgrid(typeCPU, gridCPU,w, h);
#ifdef  DEBUG

    printf("Printing Types:\n");
    for (size_t y = 0; y < hh; y++)
    {
        for (size_t x = 0; x < ww; x++)
        {
            printf("%d\t", typeCPU[y * ww + x]);
        }
        printf("\n");
    }
#endif //  
    printf("gridCPU: %f\n", gridCPU[0][5]);
    // fill it with cuda references
    cudaMalloc((void**)&gpuDataCPU->type, arrSize * sizeof(*gpuDataCPU->type)); gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(gpuDataCPU->type, typeCPU, arrSize * sizeof(CellType), cudaMemcpyHostToDevice); gpuErrchk(cudaPeekAtLastError());
    cudaMalloc((void**)&gpuDataCPU->velY, arrSize * sizeof(*gpuDataCPU->velY)); gpuErrchk(cudaPeekAtLastError());
    cudaMalloc((void**)&gpuDataCPU->velX, arrSize * sizeof(*gpuDataCPU->velX)); gpuErrchk(cudaPeekAtLastError());
    cudaMalloc((void**)&gpuDataCPU->forceY, arrSize * sizeof(*gpuDataCPU->forceY)); gpuErrchk(cudaPeekAtLastError());
    cudaMalloc((void**)&gpuDataCPU->forceX, arrSize * sizeof(*gpuDataCPU->forceX)); gpuErrchk(cudaPeekAtLastError());
    cudaMalloc((void**)&gpuDataCPU->density, arrSize * sizeof(*gpuDataCPU->density)); gpuErrchk(cudaPeekAtLastError());
    for (size_t i = 0; i < 9; i++)
    {
        cudaMalloc((void**)&gpuDataCPU->grid[i], arrSize * sizeof(*gpuDataCPU->grid[i])); gpuErrchk(cudaPeekAtLastError());
        cudaMalloc((void**)&gpuDataCPU->grid_tmp[i], arrSize * sizeof(*gpuDataCPU->grid_tmp[i])); gpuErrchk(cudaPeekAtLastError());
        cudaMemcpy(gpuDataCPU->grid[i], gridCPU[i], arrSize * sizeof(PRECISION), cudaMemcpyHostToDevice); gpuErrchk(cudaPeekAtLastError());
        cudaMemcpy(gpuDataCPU->grid_tmp[i], gridCPU[i], arrSize * sizeof(PRECISION), cudaMemcpyHostToDevice); gpuErrchk(cudaPeekAtLastError());
    }
    //int *cIdxCPU = (int*)malloc(9*sizeof(int));
    for (size_t i = 0; i < 9; i++)
    {
        gpuDataCPU->cIdx[i] = cyCPU[i] * gpuDataCPU->w + cxCPU[i];
    }
    //cudaMalloc((void**)&cIdx, 9 * sizeof(int)); gpuErrchk(cudaPeekAtLastError());
    //cudaThreadSynchronize(); gpuErrchk(cudaPeekAtLastError());
    //cudaMemcpy(cIdx, cIdxCPU, 9 * sizeof(int), cudaMemcpyHostToDevice); gpuErrchk(cudaPeekAtLastError());
#ifdef DEBUG
    printf("Printing cIdx:\n");
    for (size_t y = 0; y < 3; y++)
    {
        for (size_t x = 0; x < 3; x++)
        {
            printf("%d\t", gpuDataCPU->cIdx[y * 3 + x]);
        }
        printf("\n");
    }
    printf("Printing cIdx Inversed:\n");
    DIR DirInverseCPU[9] = { C,E,W,S,N,SE,SW,NE,NW };
    for (size_t y = 0; y < 3; y++)
    {
        for (size_t x = 0; x < 3; x++)
        {
            printf("%d\t", gpuDataCPU->cIdx[DirInverseCPU[y * 3 + x]]);
        }
        printf("\n");
    }
#endif // DEBUG


    // alloc cuda version of struct
    cudaMalloc((void**)&gpuDataLOCAL, sizeof(GpuData)); gpuErrchk(cudaPeekAtLastError());
    // copy references to cuda version
    cudaMemcpy(gpuDataLOCAL, gpuDataCPU, sizeof(GpuData), cudaMemcpyHostToDevice); gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize(); gpuErrchk(cudaPeekAtLastError());
    //free(typeCPU);
    for (size_t i = 0; i < 9; i++)
    {
       // free(gridCPU[i]);
    }
    

}

void destroy() {
    cudaFree(gpuDataCPU->type);
    cudaFree(gpuDataCPU->velY);
    cudaFree(gpuDataCPU->velX);
    cudaFree(gpuDataCPU->forceY);
    cudaFree(gpuDataCPU->forceX);
    cudaFree(gpuDataCPU->density);
    for (size_t i = 0; i < 9; i++)
    {
        cudaFree(gpuDataCPU->grid[i]);
        cudaFree(gpuDataCPU->grid_tmp[i]);
    }

    free(gpuDataCPU);
    cudaFree(gpuDataLOCAL);
}