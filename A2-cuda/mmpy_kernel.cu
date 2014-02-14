#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
using namespace std;

#if BLOCKDIM_X < BLOCKDIM_Y
__device__ void runMatMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B, int bx, int by, int tx, int ty, int I, int I16, int J, int J16, int gx, int gy, int i) {
   
    __shared__ _DOUBLE_ a[BLOCKDIM_X][BLOCKDIM_X][2][BLOCKDIM_Y/BLOCKDIM_X];
    __shared__ _DOUBLE_ b[BLOCKDIM_X][BLOCKDIM_X][2][BLOCKDIM_Y/BLOCKDIM_X];

        _DOUBLE_ c[4] = {0};

        for (int k=0; k < gy; k++){
            a[ty][tx][0][i] = A[I*N+k*by+tx];
            a[ty][tx][1][i] = A[I16*N+k*by+tx];
            b[ty][tx][0][i] = B[J+N*(k*bx+ty)];
            b[ty][tx][1][i] = B[J16+N*(k*bx+ty)];
            __syncthreads();

            for (int kk=0; kk< bx; kk++) {
                c[0] += a[ty][kk][0][i]*b[kk][tx][0][i];
                c[1] += a[ty][kk][1][i]*b[kk][tx][0][i];
                c[2] += a[ty][kk][0][i]*b[kk][tx][1][i];
                c[3] += a[ty][kk][1][i]*b[kk][tx][1][i];
            }

            __syncthreads();
        }
        C[I * N + J] = c[0];
        C[I16 * N + J] = c[1];
        C[I * N + J16] = c[2];
        C[I16 * N + J16] = c[3];
}

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

        unsigned int bx = blockDim.x, by = blockDim.y;
        unsigned int tx = threadIdx.x, ty = threadIdx.y % bx;
        unsigned int i = threadIdx.y/BLOCKDIM_X;

        unsigned int I = blockIdx.y*by + ty + i*BLOCKDIM_X, J = blockIdx.x*bx + tx;
        unsigned int I16 = (blockIdx.y+gridDim.y)*by + i*BLOCKDIM_X + ty;
        unsigned int J16 = (blockIdx.x+gridDim.x)*bx + tx;
        by=bx;

        unsigned int gx = gridDim.x, gy =  N/by; //gridDim.y * 2;
        
        runMatMul(N, C, A, B, bx, by, tx, ty, I, I16, J, J16, gx, gy, i);
}

#endif

#if BLOCKDIM_Y < BLOCKDIM_X
__device__ void runMatMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B, int bx, int by, int tx, int ty, int I, int I16, int J, int J16, int gx, int gy, int i) {

    __shared__ _DOUBLE_ a[BLOCKDIM_Y][BLOCKDIM_Y][2][BLOCKDIM_X/BLOCKDIM_Y];
    __shared__ _DOUBLE_ b[BLOCKDIM_Y][BLOCKDIM_Y][2][BLOCKDIM_X/BLOCKDIM_Y];

        _DOUBLE_ c[4] = {0};

        for (int k=0; k < gy; k++){
            a[ty][tx][0][i] = A[I*N+k*by+tx];
            a[ty][tx][1][i] = A[I16*N+k*by+tx];
            b[ty][tx][0][i] = B[J+N*(k*bx+ty)];
            b[ty][tx][1][i] = B[J16+N*(k*bx+ty)];
            __syncthreads();

            for (int kk=0; kk< bx; kk++) {
                c[0] += a[ty][kk][0][i]*b[kk][tx][0][i];
                c[1] += a[ty][kk][1][i]*b[kk][tx][0][i];
                c[2] += a[ty][kk][0][i]*b[kk][tx][1][i];
                c[3] += a[ty][kk][1][i]*b[kk][tx][1][i];
            }

            __syncthreads();
        }
        C[I * N + J] = c[0];
        C[I16 * N + J] = c[1];
        C[I * N + J16] = c[2];
        C[I16 * N + J16] = c[3];
}

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

        unsigned int bx = blockDim.x, by = blockDim.y;
        unsigned int tx = threadIdx.x % by, ty = threadIdx.y;
        unsigned int i = threadIdx.x/BLOCKDIM_Y;

        unsigned int I = blockIdx.y*by + ty, J = blockIdx.x*bx + tx + i*BLOCKDIM_Y;
        unsigned int I16 = (blockIdx.y+gridDim.y)*by + ty;
        unsigned int J16 = (blockIdx.x+gridDim.x)*bx + tx + i*BLOCKDIM_Y;
        bx=by;

        unsigned int gx = gridDim.x, gy =  N/by; //gridDim.y * 2;

        runMatMul(N, C, A, B, bx, by, tx, ty, I, I16, J, J16, gx, gy, i);
}
#endif

#if BLOCKDIM_X == BLOCKDIM_Y
__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

    const unsigned int bx = BLOCKDIM_X, by = BLOCKDIM_Y;
    const unsigned int tx = threadIdx.x, ty = threadIdx.y;
    const unsigned int I = blockIdx.y*by + ty, J = blockIdx.x*bx + tx;
    const unsigned int I16 = (blockIdx.y+gridDim.y)*by + ty;
    const unsigned int gx = gridDim.x, gy = gridDim.y * 2;
    const unsigned int J16 = (blockIdx.x+gridDim.x)*bx + tx;

    __shared__ _DOUBLE_ a[BLOCKDIM_X][BLOCKDIM_Y][2];
    __shared__ _DOUBLE_ b[BLOCKDIM_X][BLOCKDIM_Y][2];

        _DOUBLE_ c[4] = {0};

        for (int k=0; k < gy; k++){
            a[ty][tx][0] = A[I*N+k*by+tx];
            a[ty][tx][1] = A[I16*N+k*by+tx];
            b[ty][tx][0] = B[J+N*(k*bx+ty)];
            b[ty][tx][1] = B[J16+N*(k*bx+ty)];
            __syncthreads();

            for (int kk=0; kk< bx; kk++) {
                c[0] += a[ty][kk][0]*b[kk][tx][0];
                c[1] += a[ty][kk][1]*b[kk][tx][0];
                c[2] += a[ty][kk][0]*b[kk][tx][1];
                c[3] += a[ty][kk][1]*b[kk][tx][1];
            }

            __syncthreads();
        }
        C[I * N + J] = c[0];
        C[I16 * N + J] = c[1];
        C[I * N + J16] = c[2];
        C[I16 * N + J16] = c[3];
}
#endif

