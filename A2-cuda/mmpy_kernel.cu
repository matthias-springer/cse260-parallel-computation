// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
using namespace std;

#define BLOCKDIM_X 4
#define BLOCKDIM_Y 4
#define BLOCK_X 4
#define BLOCK_Y 4

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

    const unsigned int bx = BLOCK_X, by = BLOCK_Y; 
    const unsigned int tx = threadIdx.x, ty = threadIdx.y; 
    const unsigned int J = blockIdx.y*by + ty, I = blockIdx.x*bx + tx;
    const unsigned int I16 = (blockIdx.y+gridDim.y)*by + ty;
    const unsigned int gx = gridDim.x, gy = gridDim.y; 

    __shared__ _DOUBLE_ a[BLOCK_X][BLOCK_Y];
    __shared__ _DOUBLE_ b[BLOCK_X][BLOCK_Y];

    if((I < N) && (J < N)){
        _DOUBLE_ c;
         for (unsigned int k=0; k < gy; k++){ 
            a[tx][ty] = A[ I*N+k*by+ty]; 
            b[ty][tx] = B[J+N*(k*bx+tx)]; 
            __syncthreads(); // Synchronizes all threads in a block 
            for (unsigned int kk=0; kk< bx; kk++) 
                c += a[kk][tx]*b[kk][ty]; 
              
          __syncthreads(); // Avoids memory hazards 
        } 
        C[I*N+J] = c;    
    }
}
