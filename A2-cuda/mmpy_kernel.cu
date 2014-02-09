// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
using namespace std;

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16
#define BLOCK_X 16
#define BLOCK_Y 16

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

    const unsigned int bx = BLOCK_X, by = BLOCK_Y; 
    const unsigned int tx = threadIdx.x, ty = threadIdx.y; 
    const unsigned int I = blockIdx.y*by + ty, J = blockIdx.x*bx + tx;
    const unsigned int I16 = (blockIdx.y+gridDim.y)*by + ty;
    const unsigned int gx = gridDim.x, gy = gridDim.y * 2; 

    __shared__ _DOUBLE_ a[BLOCK_X][BLOCK_Y][2];
    __shared__ _DOUBLE_ b[BLOCK_X][BLOCK_Y];

    if((I < N) && (J < N)){
        _DOUBLE_ c[2] = {0,0};
        
        for (int k=0; k < gy; k++){ 
            a[ty][tx][0] = A[I*N+k*by+tx]; 
            a[ty][tx][1] = A[I16*N+k*by+tx];
            b[ty][tx] = B[J+N*(k*bx+ty)]; 
            __syncthreads(); 
            
            for (int kk=0; kk< bx; kk++) { 
                c[0] += a[ty][kk][0]*b[kk][tx];
                c[1] += a[ty][kk][1]*b[kk][tx];
            }
            
            __syncthreads();  
        }
        C[I * N + J] = c[0];
        C[I16 * N + J] = c[1];
    }
}
