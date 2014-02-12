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
    const unsigned int I0 = blockIdx.y*by + ty;
    const unsigned int J0 = blockIdx.x*bx + tx;
    const unsigned int I1 = (blockIdx.y+gridDim.y)*by + ty;
    const unsigned int gx = gridDim.x, gy = gridDim.y * 8;
    const unsigned int J1 = (blockIdx.x+gridDim.x)*bx + tx;
    const unsigned int I2 = (blockIdx.y+2*gridDim.y)*by + ty;
    const unsigned int I3 = (blockIdx.y+3*gridDim.y)*by + ty;
    const unsigned int I4 = (blockIdx.y+4*gridDim.y)*by + ty;
    const unsigned int I5 = (blockIdx.y+5*gridDim.y)*by + ty;
    const unsigned int I6 = (blockIdx.y+6*gridDim.y)*by + ty;
    const unsigned int I7 = (blockIdx.y+7*gridDim.y)*by + ty;
    const unsigned int J2 = (blockIdx.x+2*gridDim.x)*bx + tx;
    const unsigned int J3 = (blockIdx.x+3*gridDim.x)*bx + tx;
    const unsigned int J4 = (blockIdx.x+4*gridDim.x)*bx + tx;
    const unsigned int J5 = (blockIdx.x+5*gridDim.x)*bx + tx;
    const unsigned int J6 = (blockIdx.x+6*gridDim.x)*bx + tx;
    const unsigned int J7 = (blockIdx.x+7*gridDim.x)*bx + tx;

    __shared__ _DOUBLE_ a[BLOCK_X][BLOCK_Y][8];
    __shared__ _DOUBLE_ b[BLOCK_X][BLOCK_Y][8];

        _DOUBLE_ c[8][8] = {0};

        for (int aa = 0; aa<8; aa++) 
            for (int bb = 0; bb <8; bb++)
                c[aa][bb] = 0;

        for (int k=0; k < gy; k++){
/*            a[ty][tx][0] = A[I0*N+k*by+tx];
            a[ty][tx][1] = A[I1*N+k*by+tx];
            b[ty][tx][0] = B[J0+N*(k*bx+ty)];
            b[ty][tx][1] = B[J1+N*(k*bx+ty)]; */

a[ty][tx][0] = A[I0*N + k*by+tx];
a[ty][tx][1] = A[I1*N + k*by+tx];
a[ty][tx][2] = A[I2*N + k*by+tx];
a[ty][tx][3] = A[I3*N + k*by+tx];
a[ty][tx][4] = A[I4*N + k*by+tx];
a[ty][tx][5] = A[I5*N + k*by+tx];
a[ty][tx][6] = A[I6*N + k*by+tx];
a[ty][tx][7] = A[I7*N + k*by+tx];
b[ty][tx][0] = B[J0+N*(k*bx+ty)];
b[ty][tx][1] = B[J1+N*(k*bx+ty)];
b[ty][tx][2] = B[J2+N*(k*bx+ty)];
b[ty][tx][3] = B[J3+N*(k*bx+ty)];
b[ty][tx][4] = B[J4+N*(k*bx+ty)];
b[ty][tx][5] = B[J5+N*(k*bx+ty)];
b[ty][tx][6] = B[J6+N*(k*bx+ty)];
b[ty][tx][7] = B[J7+N*(k*bx+ty)];

            __syncthreads();

            for (int kk=0; kk< bx; kk++) {
/*                c[0] += a[ty][kk][0]*b[kk][tx][0];
                c[1] += a[ty][kk][1]*b[kk][tx][0];
                c[2] += a[ty][kk][0]*b[kk][tx][1];
                c[3] += a[ty][kk][1]*b[kk][tx][1]; */
c[0][0] += a[ty][kk][0]*b[kk][tx][0];
c[0][1] += a[ty][kk][0]*b[kk][tx][1];
c[0][2] += a[ty][kk][0]*b[kk][tx][2];
c[0][3] += a[ty][kk][0]*b[kk][tx][3];
c[0][4] += a[ty][kk][0]*b[kk][tx][4];
c[0][5] += a[ty][kk][0]*b[kk][tx][5];
c[0][6] += a[ty][kk][0]*b[kk][tx][6];
c[0][7] += a[ty][kk][0]*b[kk][tx][7];
c[1][0] += a[ty][kk][1]*b[kk][tx][0];
c[1][1] += a[ty][kk][1]*b[kk][tx][1];
c[1][2] += a[ty][kk][1]*b[kk][tx][2];
c[1][3] += a[ty][kk][1]*b[kk][tx][3];
c[1][4] += a[ty][kk][1]*b[kk][tx][4];
c[1][5] += a[ty][kk][1]*b[kk][tx][5];
c[1][6] += a[ty][kk][1]*b[kk][tx][6];
c[1][7] += a[ty][kk][1]*b[kk][tx][7];
c[2][0] += a[ty][kk][2]*b[kk][tx][0];
c[2][1] += a[ty][kk][2]*b[kk][tx][1];
c[2][2] += a[ty][kk][2]*b[kk][tx][2];
c[2][3] += a[ty][kk][2]*b[kk][tx][3];
c[2][4] += a[ty][kk][2]*b[kk][tx][4];
c[2][5] += a[ty][kk][2]*b[kk][tx][5];
c[2][6] += a[ty][kk][2]*b[kk][tx][6];
c[2][7] += a[ty][kk][2]*b[kk][tx][7];
c[3][0] += a[ty][kk][3]*b[kk][tx][0];
c[3][1] += a[ty][kk][3]*b[kk][tx][1];
c[3][2] += a[ty][kk][3]*b[kk][tx][2];
c[3][3] += a[ty][kk][3]*b[kk][tx][3];
c[3][4] += a[ty][kk][3]*b[kk][tx][4];
c[3][5] += a[ty][kk][3]*b[kk][tx][5];
c[3][6] += a[ty][kk][3]*b[kk][tx][6];
c[3][7] += a[ty][kk][3]*b[kk][tx][7];
c[4][0] += a[ty][kk][4]*b[kk][tx][0];
c[4][1] += a[ty][kk][4]*b[kk][tx][1];
c[4][2] += a[ty][kk][4]*b[kk][tx][2];
c[4][3] += a[ty][kk][4]*b[kk][tx][3];
c[4][4] += a[ty][kk][4]*b[kk][tx][4];
c[4][5] += a[ty][kk][4]*b[kk][tx][5];
c[4][6] += a[ty][kk][4]*b[kk][tx][6];
c[4][7] += a[ty][kk][4]*b[kk][tx][7];
c[5][0] += a[ty][kk][5]*b[kk][tx][0];
c[5][1] += a[ty][kk][5]*b[kk][tx][1];
c[5][2] += a[ty][kk][5]*b[kk][tx][2];
c[5][3] += a[ty][kk][5]*b[kk][tx][3];
c[5][4] += a[ty][kk][5]*b[kk][tx][4];
c[5][5] += a[ty][kk][5]*b[kk][tx][5];
c[5][6] += a[ty][kk][5]*b[kk][tx][6];
c[5][7] += a[ty][kk][5]*b[kk][tx][7];
c[6][0] += a[ty][kk][6]*b[kk][tx][0];
c[6][1] += a[ty][kk][6]*b[kk][tx][1];
c[6][2] += a[ty][kk][6]*b[kk][tx][2];
c[6][3] += a[ty][kk][6]*b[kk][tx][3];
c[6][4] += a[ty][kk][6]*b[kk][tx][4];
c[6][5] += a[ty][kk][6]*b[kk][tx][5];
c[6][6] += a[ty][kk][6]*b[kk][tx][6];
c[6][7] += a[ty][kk][6]*b[kk][tx][7];
c[7][0] += a[ty][kk][7]*b[kk][tx][0];
c[7][1] += a[ty][kk][7]*b[kk][tx][1];
c[7][2] += a[ty][kk][7]*b[kk][tx][2];
c[7][3] += a[ty][kk][7]*b[kk][tx][3];
c[7][4] += a[ty][kk][7]*b[kk][tx][4];
c[7][5] += a[ty][kk][7]*b[kk][tx][5];
c[7][6] += a[ty][kk][7]*b[kk][tx][6];
c[7][7] += a[ty][kk][7]*b[kk][tx][7];

            }

            __syncthreads();
        }
/*        C[I0 * N + J0] = c[0];
        C[I1 * N + J0] = c[1];
        C[I0 * N + J1] = c[2];
        C[I1 * N + J1] = c[3]; */
C[I0* N + J0] = c[0][0];
C[I0* N + J1] = c[0][1];
C[I0* N + J2] = c[0][2];
C[I0* N + J3] = c[0][3];
C[I0* N + J4] = c[0][4];
C[I0* N + J5] = c[0][5];
C[I0* N + J6] = c[0][6];
C[I0* N + J7] = c[0][7];
C[I1* N + J0] = c[1][0];
C[I1* N + J1] = c[1][1];
C[I1* N + J2] = c[1][2];
C[I1* N + J3] = c[1][3];
C[I1* N + J4] = c[1][4];
C[I1* N + J5] = c[1][5];
C[I1* N + J6] = c[1][6];
C[I1* N + J7] = c[1][7];
C[I2* N + J0] = c[2][0];
C[I2* N + J1] = c[2][1];
C[I2* N + J2] = c[2][2];
C[I2* N + J3] = c[2][3];
C[I2* N + J4] = c[2][4];
C[I2* N + J5] = c[2][5];
C[I2* N + J6] = c[2][6];
C[I2* N + J7] = c[2][7];
C[I3* N + J0] = c[3][0];
C[I3* N + J1] = c[3][1];
C[I3* N + J2] = c[3][2];
C[I3* N + J3] = c[3][3];
C[I3* N + J4] = c[3][4];
C[I3* N + J5] = c[3][5];
C[I3* N + J6] = c[3][6];
C[I3* N + J7] = c[3][7];
C[I4* N + J0] = c[4][0];
C[I4* N + J1] = c[4][1];
C[I4* N + J2] = c[4][2];
C[I4* N + J3] = c[4][3];
C[I4* N + J4] = c[4][4];
C[I4* N + J5] = c[4][5];
C[I4* N + J6] = c[4][6];
C[I4* N + J7] = c[4][7];
C[I5* N + J0] = c[5][0];
C[I5* N + J1] = c[5][1];
C[I5* N + J2] = c[5][2];
C[I5* N + J3] = c[5][3];
C[I5* N + J4] = c[5][4];
C[I5* N + J5] = c[5][5];
C[I5* N + J6] = c[5][6];
C[I5* N + J7] = c[5][7];
C[I6* N + J0] = c[6][0];
C[I6* N + J1] = c[6][1];
C[I6* N + J2] = c[6][2];
C[I6* N + J3] = c[6][3];
C[I6* N + J4] = c[6][4];
C[I6* N + J5] = c[6][5];
C[I6* N + J6] = c[6][6];
C[I6* N + J7] = c[6][7];
C[I7* N + J0] = c[7][0];
C[I7* N + J1] = c[7][1];
C[I7* N + J2] = c[7][2];
C[I7* N + J3] = c[7][3];
C[I7* N + J4] = c[7][4];
C[I7* N + J5] = c[7][5];
C[I7* N + J6] = c[7][6];
C[I7* N + J7] = c[7][7];
}

