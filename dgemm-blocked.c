/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

/* CSE 260 Assignment 1 
 * Andrew Conegliano, Matthias Springer
 */

#include <emmintrin.h>
// #include <pmmintrin.h>
// #include <malloc.h>
// #include <string.h>

const char* dgemm_desc = "Simple blocked dgemm.";

/* Default values for block size */
// #define BLOCK_SIZE_I 24
// #define BLOCK_SIZE_J 24
// #define BLOCK_SIZE_K 64

#define MIN_HELPER(a,b) (((a)<(b))?(a):(b))
#define min(a, b) MIN_HELPER(a, b)

#define do_block_not_unrolled(i, j, k, M, N, K) \
        do_block_1(lda, (M), (N), (K), A + (i)*lda + (k), B + (j)*lda + (k), C + (i)*lda + (j));

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
#define DO_BLOCK_EXPAND(odd_increment) \
static void do_block_##odd_increment (int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C) \
{ \
  for (int i = 0; i < M; ++i) \
    for (int j = 0; j < N; ++j) \
    { \
      double cij = C[i*lda+j]; \
      for (int k = 0; k < K; ++k) \
			{ \
				cij += A[i*lda+k] * B[j*(lda+(odd_increment))+k]; \
			} \
\
      C[i*lda+j] = cij; \
    } \
}

DO_BLOCK_EXPAND(0)
DO_BLOCK_EXPAND(1)

// Aligned buffers for A and B
double A_buffered[32*800*sizeof(double)] __attribute__((aligned(16)));
double B_global_transpose[800*800*sizeof(double)] __attribute__((aligned(16)));

#define DO_BLOCK_UNROLLED_WITHOUT_I_HELPER(odd_increment, block_size_i, block_size_j, block_size_k) \
static void do_block_unrolled_without_i_##odd_increment##_##block_size_i##_##block_size_j##_##block_size_k(int lda, double* restrict A, double* restrict B, double* restrict C) \
{ \
	for (int i = 0; i < (block_size_i); ++i) \
	{ \
		register int index_A = i*(lda + (odd_increment)); \
		register int index_C = i*lda; \
		\
		for (int j = 0; j < (block_size_j); j += 2) \
    { \
      __m128d cij_1 = _mm_setzero_pd(); \
			__m128d cij_2 = _mm_setzero_pd(); \
			register int index_B_1 = j*(lda + (odd_increment)); \
			register int index_B_2 = index_B_1 + (lda + (odd_increment)); \
 \
			for (int k = 0; k < (block_size_k); k += 4) { \
				__m128d a1 = _mm_load_pd(A + index_A); \
				index_A += 2; \
        __m128d b1 = _mm_load_pd(B + index_B_1); \
        index_B_1 += 2; \
				__m128d a2 = _mm_load_pd(A + index_A); \
				index_A += 2; \
        __m128d b2 = _mm_load_pd(B + index_B_1); \
        index_B_1 += 2; \
				cij_1 = _mm_add_pd(cij_1, _mm_add_pd(_mm_mul_pd(a1, b1), _mm_mul_pd(a2, b2))); \
 \
        __m128d b3 = _mm_load_pd(B + index_B_2); \
        index_B_2 += 2; \
        __m128d b4 = _mm_load_pd(B + index_B_2); \
        index_B_2 += 2; \
        cij_2 = _mm_add_pd(cij_2, _mm_add_pd(_mm_mul_pd(a1, b3), _mm_mul_pd(a2, b4))); \
			} \
 \
			__m128d cij_1_u = _mm_unpackhi_pd(cij_1, cij_1); \
			C[index_C] += _mm_cvtsd_f64(cij_1) + _mm_cvtsd_f64(cij_1_u); \
      __m128d cij_2_u = _mm_unpackhi_pd(cij_2, cij_2); \
      C[index_C + 1] += _mm_cvtsd_f64(cij_2) + _mm_cvtsd_f64(cij_2_u); \
 \
			index_A -= (block_size_k); \
			index_C += 2; \
		} \
	} \
}

#define DO_BLOCK_UNROLLED_WITHOUT_I_EXPAND(odd_increment, block_size_i, block_size_j, block_size_k) DO_BLOCK_UNROLLED_WITHOUT_I_HELPER(odd_increment, block_size_i, block_size_j, block_size_k)

#define DO_BLOCK_UNROLLED_HELPER(odd_increment, block_size_i, block_size_j, block_size_k) \
static void do_block_unrolled_with_i_##odd_increment##_##block_size_i##_##block_size_j##_##block_size_k(int lda, double* restrict A, double* restrict B, double* restrict C) \
{ \
	for (int i = 0; i < (block_size_i); i += 2) \
	{ \
		register int index_A_1 = i*(lda + (odd_increment)); \
		register int index_A_2 = index_A_1 + lda + (odd_increment); \
		register int index_C = i*lda; \
		\
		for (int j = 0; j < (block_size_j); j += 2) \
    { \
      __m128d cij_1 = _mm_setzero_pd(); \
			__m128d cij_2 = _mm_setzero_pd(); \
			__m128d cij_a2_1 = _mm_setzero_pd(); \
			__m128d cij_a2_2 = _mm_setzero_pd(); \
			register int index_B_1 = j*(lda + (odd_increment)); \
			register int index_B_2 = index_B_1 + (lda + (odd_increment)); \
 \
			for (int k = 0; k < (block_size_k); k += 4) { \
				__m128d a1 = _mm_load_pd(A + index_A_1); \
				index_A_1 += 2; \
        __m128d b1 = _mm_load_pd(B + index_B_1); \
        index_B_1 += 2; \
				__m128d a2 = _mm_load_pd(A + index_A_1); \
				index_A_1 += 2; \
        __m128d b2 = _mm_load_pd(B + index_B_1); \
        index_B_1 += 2; \
				cij_1 = _mm_add_pd(cij_1, _mm_add_pd(_mm_mul_pd(a1, b1), _mm_mul_pd(a2, b2))); \
 \
        __m128d b3 = _mm_load_pd(B + index_B_2); \
        index_B_2 += 2; \
        __m128d b4 = _mm_load_pd(B + index_B_2); \
        index_B_2 += 2; \
        cij_2 = _mm_add_pd(cij_2, _mm_add_pd(_mm_mul_pd(a1, b3), _mm_mul_pd(a2, b4))); \
 \
        __m128d a3 = _mm_load_pd(A + index_A_2); \
        index_A_2 += 2; \
        __m128d a4 = _mm_load_pd(A + index_A_2); \
        index_A_2 += 2; \
				cij_a2_1 = _mm_add_pd(cij_a2_1, _mm_add_pd(_mm_mul_pd(a3, b1), _mm_mul_pd(a4, b2))); \
				cij_a2_2 = _mm_add_pd(cij_a2_2, _mm_add_pd(_mm_mul_pd(a3, b3), _mm_mul_pd(a4, b4))); \
			} \
 \
			__m128d cij_1_u = _mm_unpackhi_pd(cij_1, cij_1); \
			C[index_C] += _mm_cvtsd_f64(cij_1) + _mm_cvtsd_f64(cij_1_u); \
      __m128d cij_2_u = _mm_unpackhi_pd(cij_2, cij_2); \
      C[index_C + 1] += _mm_cvtsd_f64(cij_2) + _mm_cvtsd_f64(cij_2_u); \
      cij_1_u = _mm_unpackhi_pd(cij_a2_1, cij_a2_1); \
      C[index_C + lda] += _mm_cvtsd_f64(cij_a2_1) + _mm_cvtsd_f64(cij_1_u); \
      cij_2_u = _mm_unpackhi_pd(cij_a2_2, cij_a2_2); \
      C[index_C + lda + 1] += _mm_cvtsd_f64(cij_a2_2) + _mm_cvtsd_f64(cij_2_u); \
 \
			index_A_1 -= (block_size_k); \
			index_A_2 -= (block_size_k); \
			index_C += 2; \
		} \
	} \
}

#define DO_BLOCK_UNROLLED_EXPAND(odd_increment, block_size_i, block_size_j, block_size_k) DO_BLOCK_UNROLLED_HELPER(odd_increment, block_size_i, block_size_j, block_size_k)

/* Expand for default case */
// DO_BLOCK_UNROLLED_EXPAND(0, BLOCK_SIZE_I, BLOCK_SIZE_J, BLOCK_SIZE_K)
// DO_BLOCK_UNROLLED_EXPAND(1, BLOCK_SIZE_I, BLOCK_SIZE_J, BLOCK_SIZE_K)

#define SQUARE_DGEMM_CALL_HELPER(do_block_function, odd_increment, block_size_i, block_size_j, block_size_k) square_dgemm_##do_block_function##_##odd_increment##_##block_size_i##_##block_size_j##_##block_size_k

#define SQUARE_DGEMM(do_block_function, odd_increment, block_size_i, block_size_j, block_size_k) SQUARE_DGEMM_CALL_HELPER(do_block_function, odd_increment, block_size_i, block_size_j, block_size_k)

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */ 
#define SQUARE_DGEMM_HELPER(do_block_function, odd_increment, block_size_i, block_size_j, block_size_k) \
void square_dgemm_##do_block_function##_##odd_increment##_##block_size_i##_##block_size_j##_##block_size_k(int lda, double* restrict A, double* restrict B, double* restrict C) \
{ \
	const register int fringe_start_i = lda / (block_size_i) * (block_size_i); \
	const register int fringe_start_j = lda / (block_size_j) * (block_size_j); \
	const register int fringe_start_k = lda / (block_size_k) * (block_size_k); \
 \
  for (int i = 0; i < fringe_start_i; i += (block_size_i)) { \
		for(int x = i; x < (block_size_i) + i; x++) { \
			memcpy(A_buffered + (x-i)*(lda+(odd_increment)), A + x*lda, lda*sizeof(double)); \
		} \
    \
		for (int j = 0; j < fringe_start_j; j += (block_size_j)) { \
      for (int k = 0; k < fringe_start_k; k += (block_size_k)) \
      { \
				do_block_unrolled_##do_block_function##_##odd_increment##_##block_size_i##_##block_size_j##_##block_size_k(lda, A_buffered + i%(block_size_i) * (lda+(odd_increment)) + k, B + j*(lda + (odd_increment)) + k, C + i*lda + j); \
				/* do_block_##odd_increment(lda, (block_size_i), (block_size_j), (block_size_k), A + i*lda + k, B + j*(lda+(odd_increment)) + k, C + i*lda + j); */ \
      } \
		} \
	} \
	if (fringe_start_i == lda && fringe_start_j == lda && fringe_start_k == lda) return; \
	/* For the fringe cases */ \
	/* {i} */ \
	{ \
		int i = fringe_start_i; \
		for (int j = 0; j < fringe_start_j; j += (block_size_j)) \
			for (int k = 0; k < fringe_start_k; k += (block_size_k)) \ 
			{ \
				do_block_##odd_increment(lda, lda-i, (block_size_j), (block_size_k), A + (i)*lda + (k), B + (j)*(lda+(odd_increment)) + (k), C + (i)*lda + (j)); \
			} \
	} \
 \
	/*( {j} */ \
  { \
    int j = fringe_start_j; \
    for (int i = 0; i < fringe_start_i; i += (block_size_i)) \
      for (int k = 0; k < fringe_start_k; k += (block_size_k)) \
      { \
				do_block_##odd_increment(lda, (block_size_i), lda-j, (block_size_k), A + (i)*lda + (k), B + (j)*(lda+(odd_increment)) + (k), C + (i)*lda + (j)); \
      } \
  } \
 \
	/* {k} */ \
  { \
    int k = fringe_start_k; \
    for (int i = 0; i < fringe_start_i; i += (block_size_i)) \
      for (int j = 0; j < fringe_start_j; j += (block_size_j)) \
      { \
				do_block_##odd_increment(lda, (block_size_i), (block_size_j), lda-k, A + (i)*lda + (k), B + (j)*(lda+(odd_increment)) + (k), C + (i)*lda + (j)); \
      } \
  } \
 \
	/* {i, j} */ \
	{ \
		int i = fringe_start_i; \
		int j = fringe_start_j; \
		for (int k = 0; k < fringe_start_k; k += (block_size_k)) \ 
		{ \
			do_block_##odd_increment(lda, lda-i, lda-j, (block_size_k), A + (i)*lda + (k), B + (j)*(lda+(odd_increment)) + (k), C + (i)*lda + (j)); \
		} \
	} \
 \
	/* {j, k} */ \
  { \
    int j = fringe_start_j; \
    int k = fringe_start_k; \
    for (int i = 0; i < fringe_start_i; i += (block_size_i)) \ 
    { \
			do_block_##odd_increment(lda, (block_size_i), lda-j, lda-k, A + (i)*lda + (k), B + (j)*(lda+(odd_increment)) + (k), C + (i)*lda + (j)); \
    } \
  } \
 \
	/* {i, k} */ \
  { \
    int i = fringe_start_i; \
    int k = fringe_start_k; \
    for (int j = 0; j < fringe_start_j; j += (block_size_j)) \
    { \
			do_block_##odd_increment(lda, lda-i, (block_size_j), lda-k, A + (i)*lda + (k), B + (j)*(lda+(odd_increment)) + (k), C + (i)*lda + (j)); \
    } \
  } \
 \
	/* {i, j, k} */ \
	{ \
		int i = fringe_start_i; \
		int j = fringe_start_j; \
		int k = fringe_start_k; \
		\
		do_block_##odd_increment(lda, lda-i, lda-j, lda-k, A + (i)*lda + (k), B + (j)*(lda+(odd_increment)) + (k), C + (i)*lda + (j)); \
	} \
}

#define SQUARE_DGEMM_EXPAND(do_block_function, odd_increment, block_size_i, block_size_j, block_size_k) SQUARE_DGEMM_HELPER(do_block_function, odd_increment, block_size_i, block_size_j, block_size_k)

/* Expand for default case */
// SQUARE_DGEMM_EXPAND(1, BLOCK_SIZE_I, BLOCK_SIZE_J, BLOCK_SIZE_K)
// SQUARE_DGEMM_EXPAND(0, BLOCK_SIZE_I, BLOCK_SIZE_J, BLOCK_SIZE_K)


#define TRANSPOSE_B_HELPER(odd_increment, block_size_i, block_size_j) \
void transpose_B_##odd_increment##_##block_size_i##_##block_size_j (int lda, double* B) \
{ \
const register int fringe_start_i = lda / (block_size_i) * (block_size_i); \
const register int fringe_start_j = lda / (block_size_j) * (block_size_j); \
for (int i = 0; i < fringe_start_i; i += (block_size_i)) { \
  for (int j = 0; j < fringe_start_j; j += (block_size_i)) { \
    for (int i0 = i; i0 < i + (block_size_i); ++i0) { \
			register int index_L = i0*(lda + (odd_increment)) + j; \
			register int index_R = i0 + j*lda; \
      for (int j0 = j; j0 < j + (block_size_j); ++j0) { \
        B_global_transpose[index_L] = B[index_R]; \
				index_L++; \
				index_R += lda; \
      } \
    } \
  } \
} \
if (fringe_start_i == lda && fringe_start_j == lda) return; \
 \
{ \
	for (int i = fringe_start_i; i < lda; i++) { \
		for (int j = 0; j < fringe_start_j; j++) { \
			B_global_transpose[i*(lda + (odd_increment)) + j] = B[j*lda + i]; \
		} \
	} \
} \
{ \
  for (int i = 0; i < fringe_start_i; i++) { \
    for (int j = fringe_start_j; j < lda; j++) { \
      B_global_transpose[i*(lda + (odd_increment)) + j] = B[j*lda + i]; \
    } \
  } \
} \
{ \
  for (int i = fringe_start_i; i < lda; i++) { \
    for (int j = fringe_start_j; j < lda; j++) { \
      B_global_transpose[i*(lda + (odd_increment)) + j] = B[j*lda + i]; \
    } \
  } \
} \
} 

#define TRANSPOSE_B_EXPAND(odd_increment, block_size_i, block_size_j) TRANSPOSE_B_HELPER(odd_increment, block_size_i, block_size_j)

#define DGEMM_SELECT(lda, do_block_function, odd_increment, block_size_i_j, block_size_k) \
case (lda): \
	transpose_B_##odd_increment##_##block_size_i_j##_##block_size_i_j ((lda), B); \
	square_dgemm_##do_block_function##_##odd_increment##_##block_size_i_j##_##block_size_i_j##_##block_size_k ((lda), A, B_global_transpose, C); \
	break;

/* Expand macros for several block sizes */
TRANSPOSE_B_EXPAND(0, 8, 8);
TRANSPOSE_B_EXPAND(1, 8, 8);
TRANSPOSE_B_EXPAND(0, 16, 16);
TRANSPOSE_B_EXPAND(1, 16, 16);
TRANSPOSE_B_EXPAND(0, 24, 24);
TRANSPOSE_B_EXPAND(1, 24, 24);
TRANSPOSE_B_EXPAND(0, 32, 32);
TRANSPOSE_B_EXPAND(1, 32, 32);
TRANSPOSE_B_EXPAND(0, 40, 40);
TRANSPOSE_B_EXPAND(1, 40, 40);
TRANSPOSE_B_EXPAND(0, 48, 48);
TRANSPOSE_B_EXPAND(1, 48, 48);
TRANSPOSE_B_EXPAND(0, 56, 56);
TRANSPOSE_B_EXPAND(1, 56, 56);
TRANSPOSE_B_EXPAND(0, 64, 64);
TRANSPOSE_B_EXPAND(1, 64, 64);

DO_BLOCK_UNROLLED_WITHOUT_I_EXPAND(0, 32, 32, 32)
DO_BLOCK_UNROLLED_WITHOUT_I_EXPAND(1, 32, 32, 32)
DO_BLOCK_UNROLLED_EXPAND(0, 8, 8, 8)
DO_BLOCK_UNROLLED_EXPAND(1, 8, 8, 8)
DO_BLOCK_UNROLLED_EXPAND(0, 8, 8, 16)
DO_BLOCK_UNROLLED_EXPAND(1, 8, 8, 16)
DO_BLOCK_UNROLLED_EXPAND(0, 8, 8, 24)
DO_BLOCK_UNROLLED_EXPAND(1, 8, 8, 24)
DO_BLOCK_UNROLLED_EXPAND(0, 8, 8, 32)
DO_BLOCK_UNROLLED_EXPAND(1, 8, 8, 32)
DO_BLOCK_UNROLLED_EXPAND(0, 8, 8, 40)
DO_BLOCK_UNROLLED_EXPAND(1, 8, 8, 40)
DO_BLOCK_UNROLLED_EXPAND(0, 8, 8, 48)
DO_BLOCK_UNROLLED_EXPAND(1, 8, 8, 48)
DO_BLOCK_UNROLLED_EXPAND(0, 8, 8, 56)
DO_BLOCK_UNROLLED_EXPAND(1, 8, 8, 56)
DO_BLOCK_UNROLLED_EXPAND(0, 8, 8, 64)
DO_BLOCK_UNROLLED_EXPAND(1, 8, 8, 64)
DO_BLOCK_UNROLLED_EXPAND(0, 16, 16, 8)
DO_BLOCK_UNROLLED_EXPAND(1, 16, 16, 8)
DO_BLOCK_UNROLLED_EXPAND(0, 16, 16, 16)
DO_BLOCK_UNROLLED_EXPAND(1, 16, 16, 16)
DO_BLOCK_UNROLLED_EXPAND(0, 16, 16, 24)
DO_BLOCK_UNROLLED_EXPAND(1, 16, 16, 24)
DO_BLOCK_UNROLLED_EXPAND(0, 16, 16, 32)
DO_BLOCK_UNROLLED_EXPAND(1, 16, 16, 32)
DO_BLOCK_UNROLLED_EXPAND(0, 16, 16, 40)
DO_BLOCK_UNROLLED_EXPAND(1, 16, 16, 40)
DO_BLOCK_UNROLLED_EXPAND(0, 16, 16, 48)
DO_BLOCK_UNROLLED_EXPAND(1, 16, 16, 48)
DO_BLOCK_UNROLLED_EXPAND(0, 16, 16, 56)
DO_BLOCK_UNROLLED_EXPAND(1, 16, 16, 56)
DO_BLOCK_UNROLLED_EXPAND(0, 16, 16, 64)
DO_BLOCK_UNROLLED_EXPAND(1, 16, 16, 64)
DO_BLOCK_UNROLLED_EXPAND(0, 24, 24, 8)
DO_BLOCK_UNROLLED_EXPAND(1, 24, 24, 8)
DO_BLOCK_UNROLLED_EXPAND(0, 24, 24, 16)
DO_BLOCK_UNROLLED_EXPAND(1, 24, 24, 16)
DO_BLOCK_UNROLLED_EXPAND(0, 24, 24, 24)
DO_BLOCK_UNROLLED_EXPAND(1, 24, 24, 24)
DO_BLOCK_UNROLLED_EXPAND(0, 24, 24, 32)
DO_BLOCK_UNROLLED_EXPAND(1, 24, 24, 32)
DO_BLOCK_UNROLLED_EXPAND(0, 24, 24, 40)
DO_BLOCK_UNROLLED_EXPAND(1, 24, 24, 40)
DO_BLOCK_UNROLLED_EXPAND(0, 24, 24, 48)
DO_BLOCK_UNROLLED_EXPAND(1, 24, 24, 48)
DO_BLOCK_UNROLLED_EXPAND(0, 24, 24, 56)
DO_BLOCK_UNROLLED_EXPAND(1, 24, 24, 56)
DO_BLOCK_UNROLLED_EXPAND(0, 24, 24, 64)
DO_BLOCK_UNROLLED_EXPAND(1, 24, 24, 64)
DO_BLOCK_UNROLLED_EXPAND(0, 32, 32, 8)
DO_BLOCK_UNROLLED_EXPAND(1, 32, 32, 8)
DO_BLOCK_UNROLLED_EXPAND(0, 32, 32, 16)
DO_BLOCK_UNROLLED_EXPAND(1, 32, 32, 16)
DO_BLOCK_UNROLLED_EXPAND(0, 32, 32, 24)
DO_BLOCK_UNROLLED_EXPAND(1, 32, 32, 24)
DO_BLOCK_UNROLLED_EXPAND(0, 32, 32, 32)
DO_BLOCK_UNROLLED_EXPAND(1, 32, 32, 32)
DO_BLOCK_UNROLLED_EXPAND(0, 32, 32, 40)
DO_BLOCK_UNROLLED_EXPAND(1, 32, 32, 40)
DO_BLOCK_UNROLLED_EXPAND(0, 32, 32, 48)
DO_BLOCK_UNROLLED_EXPAND(1, 32, 32, 48)
DO_BLOCK_UNROLLED_EXPAND(0, 32, 32, 56)
DO_BLOCK_UNROLLED_EXPAND(1, 32, 32, 56)
DO_BLOCK_UNROLLED_EXPAND(0, 32, 32, 64)
DO_BLOCK_UNROLLED_EXPAND(1, 32, 32, 64)
DO_BLOCK_UNROLLED_EXPAND(0, 40, 40, 8)
DO_BLOCK_UNROLLED_EXPAND(1, 40, 40, 8)
DO_BLOCK_UNROLLED_EXPAND(0, 40, 40, 16)
DO_BLOCK_UNROLLED_EXPAND(1, 40, 40, 16)
DO_BLOCK_UNROLLED_EXPAND(0, 40, 40, 24)
DO_BLOCK_UNROLLED_EXPAND(1, 40, 40, 24)
DO_BLOCK_UNROLLED_EXPAND(0, 40, 40, 32)
DO_BLOCK_UNROLLED_EXPAND(1, 40, 40, 32)
DO_BLOCK_UNROLLED_EXPAND(0, 40, 40, 40)
DO_BLOCK_UNROLLED_EXPAND(1, 40, 40, 40)
DO_BLOCK_UNROLLED_EXPAND(0, 40, 40, 48)
DO_BLOCK_UNROLLED_EXPAND(1, 40, 40, 48)
DO_BLOCK_UNROLLED_EXPAND(0, 40, 40, 56)
DO_BLOCK_UNROLLED_EXPAND(1, 40, 40, 56)
DO_BLOCK_UNROLLED_EXPAND(0, 40, 40, 64)
DO_BLOCK_UNROLLED_EXPAND(1, 40, 40, 64)
DO_BLOCK_UNROLLED_EXPAND(0, 48, 48, 8)
DO_BLOCK_UNROLLED_EXPAND(1, 48, 48, 8)
DO_BLOCK_UNROLLED_EXPAND(0, 48, 48, 16)
DO_BLOCK_UNROLLED_EXPAND(1, 48, 48, 16)
DO_BLOCK_UNROLLED_EXPAND(0, 48, 48, 24)
DO_BLOCK_UNROLLED_EXPAND(1, 48, 48, 24)
DO_BLOCK_UNROLLED_EXPAND(0, 48, 48, 32)
DO_BLOCK_UNROLLED_EXPAND(1, 48, 48, 32)
DO_BLOCK_UNROLLED_EXPAND(0, 48, 48, 40)
DO_BLOCK_UNROLLED_EXPAND(1, 48, 48, 40)
DO_BLOCK_UNROLLED_EXPAND(0, 48, 48, 48)
DO_BLOCK_UNROLLED_EXPAND(1, 48, 48, 48)
DO_BLOCK_UNROLLED_EXPAND(0, 48, 48, 56)
DO_BLOCK_UNROLLED_EXPAND(1, 48, 48, 56)
DO_BLOCK_UNROLLED_EXPAND(0, 48, 48, 64)
DO_BLOCK_UNROLLED_EXPAND(1, 48, 48, 64)
DO_BLOCK_UNROLLED_EXPAND(0, 56, 56, 8)
DO_BLOCK_UNROLLED_EXPAND(1, 56, 56, 8)
DO_BLOCK_UNROLLED_EXPAND(0, 56, 56, 16)
DO_BLOCK_UNROLLED_EXPAND(1, 56, 56, 16)
DO_BLOCK_UNROLLED_EXPAND(0, 56, 56, 24)
DO_BLOCK_UNROLLED_EXPAND(1, 56, 56, 24)
DO_BLOCK_UNROLLED_EXPAND(0, 56, 56, 32)
DO_BLOCK_UNROLLED_EXPAND(1, 56, 56, 32)
DO_BLOCK_UNROLLED_EXPAND(0, 56, 56, 40)
DO_BLOCK_UNROLLED_EXPAND(1, 56, 56, 40)
DO_BLOCK_UNROLLED_EXPAND(0, 56, 56, 48)
DO_BLOCK_UNROLLED_EXPAND(1, 56, 56, 48)
DO_BLOCK_UNROLLED_EXPAND(0, 56, 56, 56)
DO_BLOCK_UNROLLED_EXPAND(1, 56, 56, 56)
DO_BLOCK_UNROLLED_EXPAND(0, 56, 56, 64)
DO_BLOCK_UNROLLED_EXPAND(1, 56, 56, 64)
DO_BLOCK_UNROLLED_EXPAND(0, 64, 64, 8)
DO_BLOCK_UNROLLED_EXPAND(1, 64, 64, 8)
DO_BLOCK_UNROLLED_EXPAND(0, 64, 64, 16)
DO_BLOCK_UNROLLED_EXPAND(1, 64, 64, 16)
DO_BLOCK_UNROLLED_EXPAND(0, 64, 64, 24)
DO_BLOCK_UNROLLED_EXPAND(1, 64, 64, 24)
DO_BLOCK_UNROLLED_EXPAND(0, 64, 64, 32)
DO_BLOCK_UNROLLED_EXPAND(1, 64, 64, 32)
DO_BLOCK_UNROLLED_EXPAND(0, 64, 64, 40)
DO_BLOCK_UNROLLED_EXPAND(1, 64, 64, 40)
DO_BLOCK_UNROLLED_EXPAND(0, 64, 64, 48)
DO_BLOCK_UNROLLED_EXPAND(1, 64, 64, 48)
DO_BLOCK_UNROLLED_EXPAND(0, 64, 64, 56)
DO_BLOCK_UNROLLED_EXPAND(1, 64, 64, 56)
DO_BLOCK_UNROLLED_EXPAND(0, 64, 64, 64)
DO_BLOCK_UNROLLED_EXPAND(1, 64, 64, 64)

SQUARE_DGEMM_EXPAND(without_i, 0, 32, 32, 32)
SQUARE_DGEMM_EXPAND(without_i, 1, 32, 32, 32)
SQUARE_DGEMM_EXPAND(with_i, 0, 8, 8, 8)
SQUARE_DGEMM_EXPAND(with_i, 1, 8, 8, 8)
SQUARE_DGEMM_EXPAND(with_i, 0, 8, 8, 16)
SQUARE_DGEMM_EXPAND(with_i, 1, 8, 8, 16)
SQUARE_DGEMM_EXPAND(with_i, 0, 8, 8, 24)
SQUARE_DGEMM_EXPAND(with_i, 1, 8, 8, 24)
SQUARE_DGEMM_EXPAND(with_i, 0, 8, 8, 32)
SQUARE_DGEMM_EXPAND(with_i, 1, 8, 8, 32)
SQUARE_DGEMM_EXPAND(with_i, 0, 8, 8, 40)
SQUARE_DGEMM_EXPAND(with_i, 1, 8, 8, 40)
SQUARE_DGEMM_EXPAND(with_i, 0, 8, 8, 48)
SQUARE_DGEMM_EXPAND(with_i, 1, 8, 8, 48)
SQUARE_DGEMM_EXPAND(with_i, 0, 8, 8, 56)
SQUARE_DGEMM_EXPAND(with_i, 1, 8, 8, 56)
SQUARE_DGEMM_EXPAND(with_i, 0, 8, 8, 64)
SQUARE_DGEMM_EXPAND(with_i, 1, 8, 8, 64)
SQUARE_DGEMM_EXPAND(with_i, 0, 16, 16, 8)
SQUARE_DGEMM_EXPAND(with_i, 1, 16, 16, 8)
SQUARE_DGEMM_EXPAND(with_i, 0, 16, 16, 16)
SQUARE_DGEMM_EXPAND(with_i, 1, 16, 16, 16)
SQUARE_DGEMM_EXPAND(with_i, 0, 16, 16, 24)
SQUARE_DGEMM_EXPAND(with_i, 1, 16, 16, 24)
SQUARE_DGEMM_EXPAND(with_i, 0, 16, 16, 32)
SQUARE_DGEMM_EXPAND(with_i, 1, 16, 16, 32)
SQUARE_DGEMM_EXPAND(with_i, 0, 16, 16, 40)
SQUARE_DGEMM_EXPAND(with_i, 1, 16, 16, 40)
SQUARE_DGEMM_EXPAND(with_i, 0, 16, 16, 48)
SQUARE_DGEMM_EXPAND(with_i, 1, 16, 16, 48)
SQUARE_DGEMM_EXPAND(with_i, 0, 16, 16, 56)
SQUARE_DGEMM_EXPAND(with_i, 1, 16, 16, 56)
SQUARE_DGEMM_EXPAND(with_i, 0, 16, 16, 64)
SQUARE_DGEMM_EXPAND(with_i, 1, 16, 16, 64)
SQUARE_DGEMM_EXPAND(with_i, 0, 24, 24, 8)
SQUARE_DGEMM_EXPAND(with_i, 1, 24, 24, 8)
SQUARE_DGEMM_EXPAND(with_i, 0, 24, 24, 16)
SQUARE_DGEMM_EXPAND(with_i, 1, 24, 24, 16)
SQUARE_DGEMM_EXPAND(with_i, 0, 24, 24, 24)
SQUARE_DGEMM_EXPAND(with_i, 1, 24, 24, 24)
SQUARE_DGEMM_EXPAND(with_i, 0, 24, 24, 32)
SQUARE_DGEMM_EXPAND(with_i, 1, 24, 24, 32)
SQUARE_DGEMM_EXPAND(with_i, 0, 24, 24, 40)
SQUARE_DGEMM_EXPAND(with_i, 1, 24, 24, 40)
SQUARE_DGEMM_EXPAND(with_i, 0, 24, 24, 48)
SQUARE_DGEMM_EXPAND(with_i, 1, 24, 24, 48)
SQUARE_DGEMM_EXPAND(with_i, 0, 24, 24, 56)
SQUARE_DGEMM_EXPAND(with_i, 1, 24, 24, 56)
SQUARE_DGEMM_EXPAND(with_i, 0, 24, 24, 64)
SQUARE_DGEMM_EXPAND(with_i, 1, 24, 24, 64)
SQUARE_DGEMM_EXPAND(with_i, 0, 32, 32, 8)
SQUARE_DGEMM_EXPAND(with_i, 1, 32, 32, 8)
SQUARE_DGEMM_EXPAND(with_i, 0, 32, 32, 16)
SQUARE_DGEMM_EXPAND(with_i, 1, 32, 32, 16)
SQUARE_DGEMM_EXPAND(with_i, 0, 32, 32, 24)
SQUARE_DGEMM_EXPAND(with_i, 1, 32, 32, 24)
SQUARE_DGEMM_EXPAND(with_i, 0, 32, 32, 32)
SQUARE_DGEMM_EXPAND(with_i, 1, 32, 32, 32)
SQUARE_DGEMM_EXPAND(with_i, 0, 32, 32, 40)
SQUARE_DGEMM_EXPAND(with_i, 1, 32, 32, 40)
SQUARE_DGEMM_EXPAND(with_i, 0, 32, 32, 48)
SQUARE_DGEMM_EXPAND(with_i, 1, 32, 32, 48)
SQUARE_DGEMM_EXPAND(with_i, 0, 32, 32, 56)
SQUARE_DGEMM_EXPAND(with_i, 1, 32, 32, 56)
SQUARE_DGEMM_EXPAND(with_i, 0, 32, 32, 64)
SQUARE_DGEMM_EXPAND(with_i, 1, 32, 32, 64)
SQUARE_DGEMM_EXPAND(with_i, 0, 40, 40, 8)
SQUARE_DGEMM_EXPAND(with_i, 1, 40, 40, 8)
SQUARE_DGEMM_EXPAND(with_i, 0, 40, 40, 16)
SQUARE_DGEMM_EXPAND(with_i, 1, 40, 40, 16)
SQUARE_DGEMM_EXPAND(with_i, 0, 40, 40, 24)
SQUARE_DGEMM_EXPAND(with_i, 1, 40, 40, 24)
SQUARE_DGEMM_EXPAND(with_i, 0, 40, 40, 32)
SQUARE_DGEMM_EXPAND(with_i, 1, 40, 40, 32)
SQUARE_DGEMM_EXPAND(with_i, 0, 40, 40, 40)
SQUARE_DGEMM_EXPAND(with_i, 1, 40, 40, 40)
SQUARE_DGEMM_EXPAND(with_i, 0, 40, 40, 48)
SQUARE_DGEMM_EXPAND(with_i, 1, 40, 40, 48)
SQUARE_DGEMM_EXPAND(with_i, 0, 40, 40, 56)
SQUARE_DGEMM_EXPAND(with_i, 1, 40, 40, 56)
SQUARE_DGEMM_EXPAND(with_i, 0, 40, 40, 64)
SQUARE_DGEMM_EXPAND(with_i, 1, 40, 40, 64)
SQUARE_DGEMM_EXPAND(with_i, 0, 48, 48, 8)
SQUARE_DGEMM_EXPAND(with_i, 1, 48, 48, 8)
SQUARE_DGEMM_EXPAND(with_i, 0, 48, 48, 16)
SQUARE_DGEMM_EXPAND(with_i, 1, 48, 48, 16)
SQUARE_DGEMM_EXPAND(with_i, 0, 48, 48, 24)
SQUARE_DGEMM_EXPAND(with_i, 1, 48, 48, 24)
SQUARE_DGEMM_EXPAND(with_i, 0, 48, 48, 32)
SQUARE_DGEMM_EXPAND(with_i, 1, 48, 48, 32)
SQUARE_DGEMM_EXPAND(with_i, 0, 48, 48, 40)
SQUARE_DGEMM_EXPAND(with_i, 1, 48, 48, 40)
SQUARE_DGEMM_EXPAND(with_i, 0, 48, 48, 48)
SQUARE_DGEMM_EXPAND(with_i, 1, 48, 48, 48)
SQUARE_DGEMM_EXPAND(with_i, 0, 48, 48, 56)
SQUARE_DGEMM_EXPAND(with_i, 1, 48, 48, 56)
SQUARE_DGEMM_EXPAND(with_i, 0, 48, 48, 64)
SQUARE_DGEMM_EXPAND(with_i, 1, 48, 48, 64)
SQUARE_DGEMM_EXPAND(with_i, 0, 56, 56, 8)
SQUARE_DGEMM_EXPAND(with_i, 1, 56, 56, 8)
SQUARE_DGEMM_EXPAND(with_i, 0, 56, 56, 16)
SQUARE_DGEMM_EXPAND(with_i, 1, 56, 56, 16)
SQUARE_DGEMM_EXPAND(with_i, 0, 56, 56, 24)
SQUARE_DGEMM_EXPAND(with_i, 1, 56, 56, 24)
SQUARE_DGEMM_EXPAND(with_i, 0, 56, 56, 32)
SQUARE_DGEMM_EXPAND(with_i, 1, 56, 56, 32)
SQUARE_DGEMM_EXPAND(with_i, 0, 56, 56, 40)
SQUARE_DGEMM_EXPAND(with_i, 1, 56, 56, 40)
SQUARE_DGEMM_EXPAND(with_i, 0, 56, 56, 48)
SQUARE_DGEMM_EXPAND(with_i, 1, 56, 56, 48)
SQUARE_DGEMM_EXPAND(with_i, 0, 56, 56, 56)
SQUARE_DGEMM_EXPAND(with_i, 1, 56, 56, 56)
SQUARE_DGEMM_EXPAND(with_i, 0, 56, 56, 64)
SQUARE_DGEMM_EXPAND(with_i, 1, 56, 56, 64)
SQUARE_DGEMM_EXPAND(with_i, 0, 64, 64, 8)
SQUARE_DGEMM_EXPAND(with_i, 1, 64, 64, 8)
SQUARE_DGEMM_EXPAND(with_i, 0, 64, 64, 16)
SQUARE_DGEMM_EXPAND(with_i, 1, 64, 64, 16)
SQUARE_DGEMM_EXPAND(with_i, 0, 64, 64, 24)
SQUARE_DGEMM_EXPAND(with_i, 1, 64, 64, 24)
SQUARE_DGEMM_EXPAND(with_i, 0, 64, 64, 32)
SQUARE_DGEMM_EXPAND(with_i, 1, 64, 64, 32)
SQUARE_DGEMM_EXPAND(with_i, 0, 64, 64, 40)
SQUARE_DGEMM_EXPAND(with_i, 1, 64, 64, 40)
SQUARE_DGEMM_EXPAND(with_i, 0, 64, 64, 48)
SQUARE_DGEMM_EXPAND(with_i, 1, 64, 64, 48)
SQUARE_DGEMM_EXPAND(with_i, 0, 64, 64, 56)
SQUARE_DGEMM_EXPAND(with_i, 1, 64, 64, 56)
SQUARE_DGEMM_EXPAND(with_i, 0, 64, 64, 64)
SQUARE_DGEMM_EXPAND(with_i, 1, 64, 64, 64)

void square_dgemm (int lda, double* restrict A, double* restrict B, double* restrict C) {
	switch (lda) {
  DGEMM_SELECT(10, with_i, 0, 40, 40);
  DGEMM_SELECT(11, with_i, 1, 64, 8);
  DGEMM_SELECT(12, with_i, 0, 64, 8);
  DGEMM_SELECT(13, with_i, 1, 48, 8);
  DGEMM_SELECT(14, with_i, 0, 32, 8);
  DGEMM_SELECT(15, with_i, 1, 32, 8);
  DGEMM_SELECT(16, with_i, 0, 16, 16);
  DGEMM_SELECT(17, with_i, 1, 16, 16);
  DGEMM_SELECT(18, with_i, 0, 16, 16);
  DGEMM_SELECT(19, with_i, 1, 16, 16);
  DGEMM_SELECT(20, with_i, 0, 16, 16);
  DGEMM_SELECT(21, with_i, 1, 16, 16);
  DGEMM_SELECT(22, with_i, 0, 16, 16);
  DGEMM_SELECT(23, with_i, 1, 16, 16);
  DGEMM_SELECT(24, with_i, 0, 24, 24);
  DGEMM_SELECT(25, with_i, 1, 24, 24);
  DGEMM_SELECT(26, with_i, 0, 24, 24);
  DGEMM_SELECT(27, with_i, 1, 24, 8);
  DGEMM_SELECT(28, with_i, 0, 24, 8);
  DGEMM_SELECT(29, with_i, 1, 24, 8);
  DGEMM_SELECT(30, with_i, 0, 24, 8);
  DGEMM_SELECT(31, with_i, 1, 24, 8);
  DGEMM_SELECT(32, with_i, 0, 32, 32);
  DGEMM_SELECT(33, with_i, 1, 32, 16);
  DGEMM_SELECT(34, with_i, 0, 32, 16);
  DGEMM_SELECT(35, with_i, 1, 32, 16);
  DGEMM_SELECT(36, with_i, 0, 32, 16);
  DGEMM_SELECT(37, with_i, 1, 32, 16);
  DGEMM_SELECT(38, with_i, 0, 32, 16);
  DGEMM_SELECT(39, with_i, 1, 32, 16);
  DGEMM_SELECT(40, with_i, 0, 40, 40);
  DGEMM_SELECT(41, with_i, 1, 40, 40);
  DGEMM_SELECT(42, with_i, 0, 40, 40);
  DGEMM_SELECT(43, with_i, 1, 40, 8);
  DGEMM_SELECT(44, with_i, 0, 40, 8);
  DGEMM_SELECT(45, with_i, 1, 40, 8);
  DGEMM_SELECT(46, with_i, 0, 40, 8);
  DGEMM_SELECT(47, with_i, 1, 40, 8);
  DGEMM_SELECT(48, with_i, 0, 48, 24);
  DGEMM_SELECT(49, with_i, 1, 24, 16);
  DGEMM_SELECT(50, with_i, 0, 48, 16);
  DGEMM_SELECT(51, with_i, 1, 24, 16);
  DGEMM_SELECT(52, with_i, 0, 48, 16);
  DGEMM_SELECT(53, with_i, 1, 24, 16);
  DGEMM_SELECT(54, with_i, 0, 48, 16);
  DGEMM_SELECT(55, with_i, 1, 24, 16);
  DGEMM_SELECT(56, with_i, 0, 56, 56);
  DGEMM_SELECT(57, with_i, 1, 56, 56);
  DGEMM_SELECT(58, with_i, 0, 56, 56);
  DGEMM_SELECT(59, with_i, 1, 56, 56);
  DGEMM_SELECT(60, with_i, 0, 56, 56);
  DGEMM_SELECT(61, with_i, 1, 56, 56);
  DGEMM_SELECT(62, with_i, 0, 56, 56);
  DGEMM_SELECT(63, with_i, 1, 56, 56);
  DGEMM_SELECT(64, with_i, 0, 32, 64);
  DGEMM_SELECT(65, with_i, 1, 32, 64);
  DGEMM_SELECT(66, with_i, 0, 32, 64);
  DGEMM_SELECT(67, with_i, 1, 32, 64);
  DGEMM_SELECT(68, with_i, 0, 32, 64);
  DGEMM_SELECT(69, with_i, 1, 32, 64);
  DGEMM_SELECT(70, with_i, 0, 64, 16);
  DGEMM_SELECT(71, with_i, 1, 32, 16);
  DGEMM_SELECT(72, with_i, 0, 24, 24);
  DGEMM_SELECT(73, with_i, 1, 24, 64);
  DGEMM_SELECT(74, with_i, 0, 24, 64);
  DGEMM_SELECT(75, with_i, 1, 24, 64);
  DGEMM_SELECT(76, with_i, 0, 24, 64);
  DGEMM_SELECT(77, with_i, 1, 24, 64);
  DGEMM_SELECT(78, with_i, 0, 24, 64);
  DGEMM_SELECT(79, with_i, 1, 24, 8);
  DGEMM_SELECT(80, with_i, 0, 40, 40);
  DGEMM_SELECT(81, with_i, 1, 40, 16);
  DGEMM_SELECT(82, with_i, 0, 40, 16);
  DGEMM_SELECT(83, with_i, 1, 40, 16);
  DGEMM_SELECT(84, with_i, 0, 40, 16);
  DGEMM_SELECT(85, with_i, 1, 40, 16);
  DGEMM_SELECT(86, with_i, 0, 40, 16);
  DGEMM_SELECT(87, with_i, 1, 40, 16);
  DGEMM_SELECT(88, with_i, 0, 8, 8);
  DGEMM_SELECT(89, with_i, 1, 40, 8);
  DGEMM_SELECT(90, with_i, 0, 40, 8);
  DGEMM_SELECT(91, with_i, 1, 40, 8);
  DGEMM_SELECT(92, with_i, 0, 40, 8);
  DGEMM_SELECT(93, with_i, 1, 40, 8);
  DGEMM_SELECT(94, with_i, 0, 40, 8);
  DGEMM_SELECT(95, with_i, 1, 40, 8);
  DGEMM_SELECT(96, with_i, 0, 24, 32);
  DGEMM_SELECT(97, with_i, 1, 24, 32);
  DGEMM_SELECT(98, with_i, 0, 24, 32);
  DGEMM_SELECT(99, with_i, 1, 32, 16);
  DGEMM_SELECT(100, with_i, 0, 48, 16);
  DGEMM_SELECT(101, with_i, 1, 32, 16);
  DGEMM_SELECT(102, with_i, 0, 32, 16);
  DGEMM_SELECT(103, with_i, 1, 32, 16);
  DGEMM_SELECT(104, with_i, 0, 8, 8);
  DGEMM_SELECT(105, with_i, 1, 8, 48);
  DGEMM_SELECT(106, with_i, 0, 8, 48);
  DGEMM_SELECT(107, with_i, 1, 8, 16);
  DGEMM_SELECT(108, with_i, 0, 48, 8);
  DGEMM_SELECT(109, with_i, 1, 48, 8);
  DGEMM_SELECT(110, with_i, 0, 48, 8);
  DGEMM_SELECT(111, with_i, 1, 24, 8);
  DGEMM_SELECT(112, with_i, 0, 16, 56);
  DGEMM_SELECT(113, with_i, 1, 16, 56);
  DGEMM_SELECT(114, with_i, 0, 16, 56);
  DGEMM_SELECT(115, with_i, 1, 16, 56);
  DGEMM_SELECT(116, with_i, 0, 16, 56);
  DGEMM_SELECT(117, with_i, 1, 16, 56);
  DGEMM_SELECT(118, with_i, 0, 16, 56);
  DGEMM_SELECT(119, with_i, 1, 16, 56);
  DGEMM_SELECT(120, with_i, 0, 40, 56);
  DGEMM_SELECT(121, with_i, 1, 40, 56);
  DGEMM_SELECT(122, with_i, 0, 40, 56);
  DGEMM_SELECT(123, with_i, 1, 24, 56);
  DGEMM_SELECT(124, with_i, 0, 40, 56);
  DGEMM_SELECT(125, with_i, 1, 24, 56);
  DGEMM_SELECT(126, with_i, 0, 24, 56);
  DGEMM_SELECT(127, with_i, 1, 24, 56);
  DGEMM_SELECT(128, with_i, 0, 16, 64);
  DGEMM_SELECT(129, with_i, 1, 32, 64);
  DGEMM_SELECT(130, with_i, 0, 16, 64);
  DGEMM_SELECT(131, with_i, 1, 32, 64);
  DGEMM_SELECT(132, with_i, 0, 32, 64);
  DGEMM_SELECT(133, with_i, 1, 32, 64);
  DGEMM_SELECT(134, with_i, 0, 32, 64);
  DGEMM_SELECT(135, with_i, 1, 32, 64);
  DGEMM_SELECT(136, with_i, 0, 8, 64);
  DGEMM_SELECT(137, with_i, 1, 8, 64);
  DGEMM_SELECT(138, with_i, 0, 8, 64);
  DGEMM_SELECT(139, with_i, 1, 8, 64);
  DGEMM_SELECT(140, with_i, 0, 8, 64);
  DGEMM_SELECT(141, with_i, 1, 8, 64);
  DGEMM_SELECT(142, with_i, 0, 8, 64);
  DGEMM_SELECT(143, with_i, 1, 8, 64);
  DGEMM_SELECT(144, with_i, 0, 48, 24);
  DGEMM_SELECT(145, with_i, 1, 24, 16);
  DGEMM_SELECT(146, with_i, 0, 48, 16);
  DGEMM_SELECT(147, with_i, 1, 48, 16);
  DGEMM_SELECT(148, with_i, 0, 48, 16);
  DGEMM_SELECT(149, with_i, 1, 48, 16);
  DGEMM_SELECT(150, with_i, 0, 48, 16);
  DGEMM_SELECT(151, with_i, 1, 48, 16);
  DGEMM_SELECT(152, with_i, 0, 8, 64);
  DGEMM_SELECT(153, with_i, 1, 8, 48);
  DGEMM_SELECT(154, with_i, 0, 48, 16);
  DGEMM_SELECT(155, with_i, 1, 24, 16);
  DGEMM_SELECT(156, with_i, 0, 48, 16);
  DGEMM_SELECT(157, with_i, 1, 24, 16);
  DGEMM_SELECT(158, with_i, 0, 48, 16);
  DGEMM_SELECT(159, with_i, 1, 24, 16);
  DGEMM_SELECT(160, with_i, 0, 32, 32);
  DGEMM_SELECT(161, with_i, 1, 40, 40);
  DGEMM_SELECT(162, with_i, 0, 40, 16);
  DGEMM_SELECT(163, with_i, 1, 40, 16);
  DGEMM_SELECT(164, with_i, 0, 40, 16);
  DGEMM_SELECT(165, with_i, 1, 40, 16);
  DGEMM_SELECT(166, with_i, 0, 40, 16);
  DGEMM_SELECT(167, with_i, 1, 32, 16);
  DGEMM_SELECT(168, with_i, 0, 24, 56);
  DGEMM_SELECT(169, with_i, 1, 8, 56);
  DGEMM_SELECT(170, with_i, 0, 8, 56);
  DGEMM_SELECT(171, with_i, 1, 8, 56);
  DGEMM_SELECT(172, with_i, 0, 8, 56);
  DGEMM_SELECT(173, with_i, 1, 24, 56);
  DGEMM_SELECT(174, with_i, 0, 24, 56);
  DGEMM_SELECT(175, with_i, 1, 24, 56);
  DGEMM_SELECT(176, with_i, 0, 16, 56);
  DGEMM_SELECT(177, with_i, 1, 16, 56);
  DGEMM_SELECT(178, with_i, 0, 16, 56);
  DGEMM_SELECT(179, with_i, 1, 16, 56);
  DGEMM_SELECT(180, with_i, 0, 16, 56);
  DGEMM_SELECT(181, with_i, 1, 16, 56);
  DGEMM_SELECT(182, with_i, 0, 16, 56);
  DGEMM_SELECT(183, with_i, 1, 16, 56);
  DGEMM_SELECT(184, with_i, 0, 8, 56);
  DGEMM_SELECT(185, with_i, 1, 8, 56);
  DGEMM_SELECT(186, with_i, 0, 8, 56);
  DGEMM_SELECT(187, with_i, 1, 8, 56);
  DGEMM_SELECT(188, with_i, 0, 8, 56);
  DGEMM_SELECT(189, with_i, 1, 8, 56);
  DGEMM_SELECT(190, with_i, 0, 8, 56);
  DGEMM_SELECT(191, with_i, 1, 8, 56);
  DGEMM_SELECT(192, with_i, 0, 24, 64);
  DGEMM_SELECT(193, with_i, 1, 32, 64);
  DGEMM_SELECT(194, with_i, 0, 24, 64);
  DGEMM_SELECT(195, with_i, 1, 32, 64);
  DGEMM_SELECT(196, with_i, 0, 24, 64);
  DGEMM_SELECT(197, with_i, 1, 32, 64);
  DGEMM_SELECT(198, with_i, 0, 32, 64);
  DGEMM_SELECT(199, with_i, 1, 32, 64);
  DGEMM_SELECT(200, with_i, 0, 40, 64);
  DGEMM_SELECT(201, with_i, 1, 40, 64);
  DGEMM_SELECT(202, with_i, 0, 40, 64);
  DGEMM_SELECT(203, with_i, 1, 8, 64);
  DGEMM_SELECT(204, with_i, 0, 8, 64);
  DGEMM_SELECT(205, with_i, 1, 40, 64);
  DGEMM_SELECT(206, with_i, 0, 40, 64);
  DGEMM_SELECT(207, with_i, 1, 40, 64);
  DGEMM_SELECT(208, with_i, 0, 16, 64);
  DGEMM_SELECT(209, with_i, 1, 16, 64);
  DGEMM_SELECT(210, with_i, 0, 16, 64);
  DGEMM_SELECT(211, with_i, 1, 16, 64);
  DGEMM_SELECT(212, with_i, 0, 16, 64);
  DGEMM_SELECT(213, with_i, 1, 16, 64);
  DGEMM_SELECT(214, with_i, 0, 16, 64);
  DGEMM_SELECT(215, with_i, 1, 16, 16);
  DGEMM_SELECT(216, with_i, 0, 24, 24);
  DGEMM_SELECT(217, with_i, 1, 24, 24);
  DGEMM_SELECT(218, with_i, 0, 24, 64);
  DGEMM_SELECT(219, with_i, 1, 24, 24);
  DGEMM_SELECT(220, with_i, 0, 24, 64);
  DGEMM_SELECT(221, with_i, 1, 24, 24);
  DGEMM_SELECT(222, with_i, 0, 24, 24);
  DGEMM_SELECT(223, with_i, 1, 24, 16);
  DGEMM_SELECT(224, with_i, 0, 32, 56);
  DGEMM_SELECT(225, with_i, 1, 32, 56);
  DGEMM_SELECT(226, with_i, 0, 32, 56);
  DGEMM_SELECT(227, with_i, 1, 32, 56);
  DGEMM_SELECT(228, with_i, 0, 32, 56);
  DGEMM_SELECT(229, with_i, 1, 32, 56);
  DGEMM_SELECT(230, with_i, 0, 32, 56);
  DGEMM_SELECT(231, with_i, 1, 32, 56);
  DGEMM_SELECT(232, with_i, 0, 8, 56);
  DGEMM_SELECT(233, with_i, 1, 8, 56);
  DGEMM_SELECT(234, with_i, 0, 8, 56);
  DGEMM_SELECT(235, with_i, 1, 8, 56);
  DGEMM_SELECT(236, with_i, 0, 8, 56);
  DGEMM_SELECT(237, with_i, 1, 8, 56);
  DGEMM_SELECT(238, with_i, 0, 8, 56);
  DGEMM_SELECT(239, with_i, 1, 8, 56);
  DGEMM_SELECT(240, with_i, 0, 40, 56);
  DGEMM_SELECT(241, with_i, 1, 40, 56);
  DGEMM_SELECT(242, with_i, 0, 40, 56);
  DGEMM_SELECT(243, with_i, 1, 40, 56);
  DGEMM_SELECT(244, with_i, 0, 40, 56);
  DGEMM_SELECT(245, with_i, 1, 24, 56);
  DGEMM_SELECT(246, with_i, 0, 24, 56);
  DGEMM_SELECT(247, with_i, 1, 40, 16);
  DGEMM_SELECT(248, with_i, 0, 8, 56);
  DGEMM_SELECT(249, with_i, 1, 8, 56);
  DGEMM_SELECT(250, with_i, 0, 8, 56);
  DGEMM_SELECT(251, with_i, 1, 8, 56);
  DGEMM_SELECT(252, with_i, 0, 8, 56);
  DGEMM_SELECT(253, with_i, 1, 40, 16);
  DGEMM_SELECT(254, with_i, 0, 8, 56);
  DGEMM_SELECT(255, with_i, 1, 8, 56);
  DGEMM_SELECT(256, with_i, 0, 8, 64);
  DGEMM_SELECT(257, with_i, 1, 8, 64);
  DGEMM_SELECT(258, with_i, 0, 8, 64);
  DGEMM_SELECT(259, with_i, 1, 16, 64);
  DGEMM_SELECT(260, with_i, 0, 16, 64);
  DGEMM_SELECT(261, with_i, 1, 32, 64);
  DGEMM_SELECT(262, with_i, 0, 32, 64);
  DGEMM_SELECT(263, with_i, 1, 32, 64);
  DGEMM_SELECT(264, with_i, 0, 24, 64);
  DGEMM_SELECT(265, with_i, 1, 24, 64);
  DGEMM_SELECT(266, with_i, 0, 24, 64);
  DGEMM_SELECT(267, with_i, 1, 24, 64);
  DGEMM_SELECT(268, with_i, 0, 24, 64);
  DGEMM_SELECT(269, with_i, 1, 24, 64);
  DGEMM_SELECT(270, with_i, 0, 24, 64);
  DGEMM_SELECT(271, with_i, 1, 24, 64);
  DGEMM_SELECT(272, with_i, 0, 16, 64);
  DGEMM_SELECT(273, with_i, 1, 16, 64);
  DGEMM_SELECT(274, with_i, 0, 16, 64);
  DGEMM_SELECT(275, with_i, 1, 16, 64);
  DGEMM_SELECT(276, with_i, 0, 16, 64);
  DGEMM_SELECT(277, with_i, 1, 16, 64);
  DGEMM_SELECT(278, with_i, 0, 16, 64);
  DGEMM_SELECT(279, with_i, 1, 16, 64);
  DGEMM_SELECT(280, with_i, 0, 40, 56);
  DGEMM_SELECT(281, with_i, 1, 40, 56);
  DGEMM_SELECT(282, with_i, 0, 40, 56);
  DGEMM_SELECT(283, with_i, 1, 40, 56);
  DGEMM_SELECT(284, with_i, 0, 40, 56);
  DGEMM_SELECT(285, with_i, 1, 40, 56);
  DGEMM_SELECT(286, with_i, 0, 40, 56);
  DGEMM_SELECT(287, with_i, 1, 40, 56);
  DGEMM_SELECT(288, with_i, 0, 32, 56);
  DGEMM_SELECT(289, with_i, 1, 24, 56);
  DGEMM_SELECT(290, with_i, 0, 32, 56);
  DGEMM_SELECT(291, with_i, 1, 24, 56);
  DGEMM_SELECT(292, with_i, 0, 32, 56);
  DGEMM_SELECT(293, with_i, 1, 24, 56);
  DGEMM_SELECT(294, with_i, 0, 32, 56);
  DGEMM_SELECT(295, with_i, 1, 24, 56);
  DGEMM_SELECT(296, with_i, 0, 8, 56);
  DGEMM_SELECT(297, with_i, 1, 8, 56);
  DGEMM_SELECT(298, with_i, 0, 8, 56);
  DGEMM_SELECT(299, with_i, 1, 8, 56);
  DGEMM_SELECT(300, with_i, 0, 8, 56);
  DGEMM_SELECT(301, with_i, 1, 8, 56);
  DGEMM_SELECT(302, with_i, 0, 8, 56);
  DGEMM_SELECT(303, with_i, 1, 8, 56);
  DGEMM_SELECT(304, with_i, 0, 16, 56);
  DGEMM_SELECT(305, with_i, 1, 16, 56);
  DGEMM_SELECT(306, with_i, 0, 16, 56);
  DGEMM_SELECT(307, with_i, 1, 16, 56);
  DGEMM_SELECT(308, with_i, 0, 16, 56);
  DGEMM_SELECT(309, with_i, 1, 16, 56);
  DGEMM_SELECT(310, with_i, 0, 16, 56);
  DGEMM_SELECT(311, with_i, 1, 16, 16);
  DGEMM_SELECT(312, with_i, 0, 24, 24);
  DGEMM_SELECT(313, with_i, 1, 24, 24);
  DGEMM_SELECT(314, with_i, 0, 24, 24);
  DGEMM_SELECT(315, with_i, 1, 24, 24);
  DGEMM_SELECT(316, with_i, 0, 24, 24);
  DGEMM_SELECT(317, with_i, 1, 24, 24);
  DGEMM_SELECT(318, with_i, 0, 24, 24);
  DGEMM_SELECT(319, with_i, 1, 24, 24);
  DGEMM_SELECT(320, with_i, 0, 16, 64);
  DGEMM_SELECT(321, with_i, 1, 32, 64);
  DGEMM_SELECT(322, with_i, 0, 32, 64);
  DGEMM_SELECT(323, with_i, 1, 32, 64);
  DGEMM_SELECT(324, with_i, 0, 32, 64);
  DGEMM_SELECT(325, with_i, 1, 32, 64);
  DGEMM_SELECT(326, with_i, 0, 32, 64);
  DGEMM_SELECT(327, with_i, 1, 32, 64);
  DGEMM_SELECT(328, with_i, 0, 8, 64);
  DGEMM_SELECT(329, with_i, 1, 8, 64);
  DGEMM_SELECT(330, with_i, 0, 8, 64);
  DGEMM_SELECT(331, with_i, 1, 8, 64);
  DGEMM_SELECT(332, with_i, 0, 8, 64);
  DGEMM_SELECT(333, with_i, 1, 8, 64);
  DGEMM_SELECT(334, with_i, 0, 8, 64);
  DGEMM_SELECT(335, with_i, 1, 8, 64);
  DGEMM_SELECT(336, with_i, 0, 24, 56);
  DGEMM_SELECT(337, with_i, 1, 24, 56);
  DGEMM_SELECT(338, with_i, 0, 24, 56);
  DGEMM_SELECT(339, with_i, 1, 16, 56);
  DGEMM_SELECT(340, with_i, 0, 16, 56);
  DGEMM_SELECT(341, with_i, 1, 16, 56);
  DGEMM_SELECT(342, with_i, 0, 16, 56);
  DGEMM_SELECT(343, with_i, 1, 24, 56);
  DGEMM_SELECT(344, with_i, 0, 8, 56);
  DGEMM_SELECT(345, with_i, 1, 8, 56);
  DGEMM_SELECT(346, with_i, 0, 8, 56);
  DGEMM_SELECT(347, with_i, 1, 8, 56);
  DGEMM_SELECT(348, with_i, 0, 8, 56);
  DGEMM_SELECT(349, with_i, 1, 8, 56);
  DGEMM_SELECT(350, with_i, 0, 8, 56);
  DGEMM_SELECT(351, with_i, 1, 8, 56);
  DGEMM_SELECT(352, with_i, 0, 32, 56);
  DGEMM_SELECT(353, with_i, 1, 32, 56);
  DGEMM_SELECT(354, with_i, 0, 32, 56);
  DGEMM_SELECT(355, with_i, 1, 32, 56);
  DGEMM_SELECT(356, with_i, 0, 32, 56);
  DGEMM_SELECT(357, with_i, 1, 32, 56);
  DGEMM_SELECT(358, with_i, 0, 32, 56);
  DGEMM_SELECT(359, with_i, 1, 32, 56);
  DGEMM_SELECT(360, with_i, 0, 40, 56);
  DGEMM_SELECT(361, with_i, 1, 24, 56);
  DGEMM_SELECT(362, with_i, 0, 40, 56);
  DGEMM_SELECT(363, with_i, 1, 24, 56);
  DGEMM_SELECT(364, with_i, 0, 40, 56);
  DGEMM_SELECT(365, with_i, 1, 40, 56);
  DGEMM_SELECT(366, with_i, 0, 40, 56);
  DGEMM_SELECT(367, with_i, 1, 40, 56);
  DGEMM_SELECT(368, with_i, 0, 16, 56);
  DGEMM_SELECT(369, with_i, 1, 16, 56);
  DGEMM_SELECT(370, with_i, 0, 16, 56);
  DGEMM_SELECT(371, with_i, 1, 16, 56);
  DGEMM_SELECT(372, with_i, 0, 16, 56);
  DGEMM_SELECT(373, with_i, 1, 16, 56);
  DGEMM_SELECT(374, with_i, 0, 16, 56);
  DGEMM_SELECT(375, with_i, 1, 16, 16);
  DGEMM_SELECT(376, with_i, 0, 8, 56);
  DGEMM_SELECT(377, with_i, 1, 16, 16);
  DGEMM_SELECT(378, with_i, 0, 8, 56);
  DGEMM_SELECT(379, with_i, 1, 16, 16);
  DGEMM_SELECT(380, with_i, 0, 8, 56);
  DGEMM_SELECT(381, with_i, 1, 40, 16);
  DGEMM_SELECT(382, with_i, 0, 40, 16);
  DGEMM_SELECT(383, with_i, 1, 16, 16);
  DGEMM_SELECT(384, with_i, 0, 16, 64);
  DGEMM_SELECT(385, with_i, 1, 16, 64);
  DGEMM_SELECT(386, with_i, 0, 24, 64);
  DGEMM_SELECT(387, with_i, 1, 32, 64);
  DGEMM_SELECT(388, with_i, 0, 24, 64);
  DGEMM_SELECT(389, with_i, 1, 32, 64);
  DGEMM_SELECT(390, with_i, 0, 24, 64);
  DGEMM_SELECT(391, with_i, 1, 32, 64);
  DGEMM_SELECT(392, with_i, 0, 8, 56);
  DGEMM_SELECT(393, with_i, 1, 8, 56);
  DGEMM_SELECT(394, with_i, 0, 8, 56);
  DGEMM_SELECT(395, with_i, 1, 8, 56);
  DGEMM_SELECT(396, with_i, 0, 8, 56);
  DGEMM_SELECT(397, with_i, 1, 56, 56);
  DGEMM_SELECT(398, with_i, 0, 8, 56);
  DGEMM_SELECT(399, with_i, 1, 56, 56);
  DGEMM_SELECT(400, with_i, 0, 40, 56);
  DGEMM_SELECT(401, with_i, 1, 40, 56);
  DGEMM_SELECT(402, with_i, 0, 40, 56);
  DGEMM_SELECT(403, with_i, 1, 40, 56);
  DGEMM_SELECT(404, with_i, 0, 40, 56);
  DGEMM_SELECT(405, with_i, 1, 40, 56);
  DGEMM_SELECT(406, with_i, 0, 40, 56);
  DGEMM_SELECT(407, with_i, 1, 16, 56);
  DGEMM_SELECT(408, with_i, 0, 24, 56);
  DGEMM_SELECT(409, with_i, 1, 24, 56);
  DGEMM_SELECT(410, with_i, 0, 24, 56);
  DGEMM_SELECT(411, with_i, 1, 24, 56);
  DGEMM_SELECT(412, with_i, 0, 24, 56);
  DGEMM_SELECT(413, with_i, 1, 24, 56);
  DGEMM_SELECT(414, with_i, 0, 24, 56);
  DGEMM_SELECT(415, with_i, 1, 24, 56);
  DGEMM_SELECT(416, with_i, 0, 16, 56);
  DGEMM_SELECT(417, with_i, 1, 32, 56);
  DGEMM_SELECT(418, with_i, 0, 32, 56);
  DGEMM_SELECT(419, with_i, 1, 32, 56);
  DGEMM_SELECT(420, with_i, 0, 32, 56);
  DGEMM_SELECT(421, with_i, 1, 32, 56);
  DGEMM_SELECT(422, with_i, 0, 32, 56);
  DGEMM_SELECT(423, with_i, 1, 32, 56);
  DGEMM_SELECT(424, with_i, 0, 8, 56);
  DGEMM_SELECT(425, with_i, 1, 8, 56);
  DGEMM_SELECT(426, with_i, 0, 8, 56);
  DGEMM_SELECT(427, with_i, 1, 8, 56);
  DGEMM_SELECT(428, with_i, 0, 8, 56);
  DGEMM_SELECT(429, with_i, 1, 8, 56);
  DGEMM_SELECT(430, with_i, 0, 8, 56);
  DGEMM_SELECT(431, with_i, 1, 32, 16);
  DGEMM_SELECT(432, with_i, 0, 48, 24);
  DGEMM_SELECT(433, with_i, 1, 48, 48);
  DGEMM_SELECT(434, with_i, 0, 48, 24);
  DGEMM_SELECT(435, with_i, 1, 48, 24);
  DGEMM_SELECT(436, with_i, 0, 48, 24);
  DGEMM_SELECT(437, with_i, 1, 48, 24);
  DGEMM_SELECT(438, with_i, 0, 48, 24);
  DGEMM_SELECT(439, with_i, 1, 48, 16);
  DGEMM_SELECT(440, with_i, 0, 40, 40);
  DGEMM_SELECT(441, with_i, 1, 40, 40);
  DGEMM_SELECT(442, with_i, 0, 40, 40);
  DGEMM_SELECT(443, with_i, 1, 40, 40);
  DGEMM_SELECT(444, with_i, 0, 40, 40);
  DGEMM_SELECT(445, with_i, 1, 40, 40);
  DGEMM_SELECT(446, with_i, 0, 40, 40);
  DGEMM_SELECT(447, with_i, 1, 40, 40);
  DGEMM_SELECT(448, with_i, 0, 16, 64);
  DGEMM_SELECT(449, with_i, 1, 32, 64);
  DGEMM_SELECT(450, with_i, 0, 32, 64);
  DGEMM_SELECT(451, with_i, 1, 32, 64);
  DGEMM_SELECT(452, with_i, 0, 32, 64);
  DGEMM_SELECT(453, with_i, 1, 32, 64);
  DGEMM_SELECT(454, with_i, 0, 32, 64);
  DGEMM_SELECT(455, with_i, 1, 32, 64);
  DGEMM_SELECT(456, with_i, 0, 24, 64);
  DGEMM_SELECT(457, with_i, 1, 24, 56);
  DGEMM_SELECT(458, with_i, 0, 24, 64);
  DGEMM_SELECT(459, with_i, 1, 24, 56);
  DGEMM_SELECT(460, with_i, 0, 24, 64);
  DGEMM_SELECT(461, with_i, 1, 24, 56);
  DGEMM_SELECT(462, with_i, 0, 24, 64);
  DGEMM_SELECT(463, with_i, 1, 24, 56);
  DGEMM_SELECT(464, with_i, 0, 16, 64);
  DGEMM_SELECT(465, with_i, 1, 16, 64);
  DGEMM_SELECT(466, with_i, 0, 16, 64);
  DGEMM_SELECT(467, with_i, 1, 16, 64);
  DGEMM_SELECT(468, with_i, 0, 16, 64);
  DGEMM_SELECT(469, with_i, 1, 16, 64);
  DGEMM_SELECT(470, with_i, 0, 16, 64);
  DGEMM_SELECT(471, with_i, 1, 16, 64);
  DGEMM_SELECT(472, with_i, 0, 8, 56);
  DGEMM_SELECT(473, with_i, 1, 8, 64);
  DGEMM_SELECT(474, with_i, 0, 8, 56);
  DGEMM_SELECT(475, with_i, 1, 8, 64);
  DGEMM_SELECT(476, with_i, 0, 8, 56);
  DGEMM_SELECT(477, with_i, 1, 8, 64);
  DGEMM_SELECT(478, with_i, 0, 8, 56);
  DGEMM_SELECT(479, with_i, 1, 8, 64);
  DGEMM_SELECT(480, with_i, 0, 16, 64);
  DGEMM_SELECT(481, with_i, 1, 32, 64);
  DGEMM_SELECT(482, with_i, 0, 16, 64);
  DGEMM_SELECT(483, with_i, 1, 32, 64);
  DGEMM_SELECT(484, with_i, 0, 16, 56);
  DGEMM_SELECT(485, with_i, 1, 24, 56);
  DGEMM_SELECT(486, with_i, 0, 16, 64);
  DGEMM_SELECT(487, with_i, 1, 24, 56);
  DGEMM_SELECT(488, with_i, 0, 8, 56);
  DGEMM_SELECT(489, with_i, 1, 8, 64);
  DGEMM_SELECT(490, with_i, 0, 8, 56);
  DGEMM_SELECT(491, with_i, 1, 8, 64);
  DGEMM_SELECT(492, with_i, 0, 8, 56);
  DGEMM_SELECT(493, with_i, 1, 32, 16);
  DGEMM_SELECT(494, with_i, 0, 8, 56);
  DGEMM_SELECT(495, with_i, 1, 32, 16);
  DGEMM_SELECT(496, with_i, 0, 16, 56);
  DGEMM_SELECT(497, with_i, 1, 16, 64);
  DGEMM_SELECT(498, with_i, 0, 16, 64);
  DGEMM_SELECT(499, with_i, 1, 16, 16);
  DGEMM_SELECT(500, with_i, 0, 16, 56);
  DGEMM_SELECT(501, with_i, 1, 16, 16);
  DGEMM_SELECT(502, with_i, 0, 16, 56);
  DGEMM_SELECT(503, with_i, 1, 16, 16);
  DGEMM_SELECT(504, with_i, 0, 8, 56);
  DGEMM_SELECT(505, with_i, 1, 8, 56);
  DGEMM_SELECT(506, with_i, 0, 8, 56);
  DGEMM_SELECT(507, with_i, 1, 8, 56);
  DGEMM_SELECT(508, with_i, 0, 8, 56);
  DGEMM_SELECT(509, with_i, 1, 8, 56);
  DGEMM_SELECT(510, with_i, 0, 8, 56);
  DGEMM_SELECT(511, with_i, 1, 56, 56);
  DGEMM_SELECT(512, with_i, 0, 16, 64);
  DGEMM_SELECT(513, with_i, 1, 8, 64);
  DGEMM_SELECT(514, with_i, 0, 8, 64);
  DGEMM_SELECT(515, with_i, 1, 8, 64);
  DGEMM_SELECT(516, with_i, 0, 8, 64);
  DGEMM_SELECT(517, with_i, 1, 8, 64);
  DGEMM_SELECT(518, with_i, 0, 8, 64);
  DGEMM_SELECT(519, with_i, 1, 8, 64);
  DGEMM_SELECT(520, with_i, 0, 8, 64);
  DGEMM_SELECT(521, with_i, 1, 40, 64);
  DGEMM_SELECT(522, with_i, 0, 40, 64);
  DGEMM_SELECT(523, with_i, 1, 40, 64);
  DGEMM_SELECT(524, with_i, 0, 40, 64);
  DGEMM_SELECT(525, with_i, 1, 40, 64);
  DGEMM_SELECT(526, with_i, 0, 40, 64);
  DGEMM_SELECT(527, with_i, 1, 40, 64);
  DGEMM_SELECT(528, with_i, 0, 24, 64);
  DGEMM_SELECT(529, with_i, 1, 16, 64);
  DGEMM_SELECT(530, with_i, 0, 24, 64);
  DGEMM_SELECT(531, with_i, 1, 16, 64);
  DGEMM_SELECT(532, with_i, 0, 24, 64);
  DGEMM_SELECT(533, with_i, 1, 16, 64);
  DGEMM_SELECT(534, with_i, 0, 24, 64);
  DGEMM_SELECT(535, with_i, 1, 16, 64);
  DGEMM_SELECT(536, with_i, 0, 8, 64);
  DGEMM_SELECT(537, with_i, 1, 8, 64);
  DGEMM_SELECT(538, with_i, 0, 8, 64);
  DGEMM_SELECT(539, with_i, 1, 8, 64);
  DGEMM_SELECT(540, with_i, 0, 8, 64);
  DGEMM_SELECT(541, with_i, 1, 8, 64);
  DGEMM_SELECT(542, with_i, 0, 16, 64);
  DGEMM_SELECT(543, with_i, 1, 8, 64);
  DGEMM_SELECT(544, with_i, 0, 32, 64);
  DGEMM_SELECT(545, with_i, 1, 32, 64);
  DGEMM_SELECT(546, with_i, 0, 32, 64);
  DGEMM_SELECT(547, with_i, 1, 32, 64);
  DGEMM_SELECT(548, with_i, 0, 16, 64);
  DGEMM_SELECT(549, with_i, 1, 32, 64);
  DGEMM_SELECT(550, with_i, 0, 32, 64);
  DGEMM_SELECT(551, with_i, 1, 32, 64);
  DGEMM_SELECT(552, with_i, 0, 24, 64);
  DGEMM_SELECT(553, with_i, 1, 24, 64);
  DGEMM_SELECT(554, with_i, 0, 24, 64);
  DGEMM_SELECT(555, with_i, 1, 24, 64);
  DGEMM_SELECT(556, with_i, 0, 24, 64);
  DGEMM_SELECT(557, with_i, 1, 24, 64);
  DGEMM_SELECT(558, with_i, 0, 24, 64);
  DGEMM_SELECT(559, with_i, 1, 24, 64);
  DGEMM_SELECT(560, with_i, 0, 40, 56);
  DGEMM_SELECT(561, with_i, 1, 40, 56);
  DGEMM_SELECT(562, with_i, 0, 40, 56);
  DGEMM_SELECT(563, with_i, 1, 40, 56);
  DGEMM_SELECT(564, with_i, 0, 40, 56);
  DGEMM_SELECT(565, with_i, 1, 40, 56);
  DGEMM_SELECT(566, with_i, 0, 40, 56);
  DGEMM_SELECT(567, with_i, 1, 40, 56);
  DGEMM_SELECT(568, with_i, 0, 8, 56);
  DGEMM_SELECT(569, with_i, 1, 40, 56);
  DGEMM_SELECT(570, with_i, 0, 8, 56);
  DGEMM_SELECT(571, with_i, 1, 8, 56);
  DGEMM_SELECT(572, with_i, 0, 8, 56);
  DGEMM_SELECT(573, with_i, 1, 40, 56);
  DGEMM_SELECT(574, with_i, 0, 8, 56);
  DGEMM_SELECT(575, with_i, 1, 8, 56);
  DGEMM_SELECT(576, with_i, 0, 24, 64);
  DGEMM_SELECT(577, with_i, 1, 32, 64);
  DGEMM_SELECT(578, with_i, 0, 32, 64);
  DGEMM_SELECT(579, with_i, 1, 32, 64);
  DGEMM_SELECT(580, with_i, 0, 32, 64);
  DGEMM_SELECT(581, with_i, 1, 32, 64);
  DGEMM_SELECT(582, with_i, 0, 32, 64);
  DGEMM_SELECT(583, with_i, 1, 32, 64);
  DGEMM_SELECT(584, with_i, 0, 32, 64);
  DGEMM_SELECT(585, with_i, 1, 32, 64);
  DGEMM_SELECT(586, with_i, 0, 32, 64);
  DGEMM_SELECT(587, with_i, 1, 32, 64);
  DGEMM_SELECT(588, with_i, 0, 32, 64);
  DGEMM_SELECT(589, with_i, 1, 32, 64);
  DGEMM_SELECT(590, with_i, 0, 32, 64);
  DGEMM_SELECT(591, with_i, 1, 32, 64);
  DGEMM_SELECT(592, with_i, 0, 16, 64);
  DGEMM_SELECT(593, with_i, 1, 16, 64);
  DGEMM_SELECT(594, with_i, 0, 16, 64);
  DGEMM_SELECT(595, with_i, 1, 16, 64);
  DGEMM_SELECT(596, with_i, 0, 16, 64);
  DGEMM_SELECT(597, with_i, 1, 16, 64);
  DGEMM_SELECT(598, with_i, 0, 16, 64);
  DGEMM_SELECT(599, with_i, 1, 16, 64);
  DGEMM_SELECT(600, with_i, 0, 40, 64);
  DGEMM_SELECT(601, with_i, 1, 40, 64);
  DGEMM_SELECT(602, with_i, 0, 40, 64);
  DGEMM_SELECT(603, with_i, 1, 40, 64);
  DGEMM_SELECT(604, with_i, 0, 40, 64);
  DGEMM_SELECT(605, with_i, 1, 40, 64);
  DGEMM_SELECT(606, with_i, 0, 40, 64);
  DGEMM_SELECT(607, with_i, 1, 24, 64);
  DGEMM_SELECT(608, with_i, 0, 32, 64);
  DGEMM_SELECT(609, with_i, 1, 32, 64);
  DGEMM_SELECT(610, with_i, 0, 32, 64);
  DGEMM_SELECT(611, with_i, 1, 32, 64);
  DGEMM_SELECT(612, with_i, 0, 32, 64);
  DGEMM_SELECT(613, with_i, 1, 32, 64);
  DGEMM_SELECT(614, with_i, 0, 32, 64);
  DGEMM_SELECT(615, with_i, 1, 32, 64);
  DGEMM_SELECT(616, with_i, 0, 8, 56);
  DGEMM_SELECT(617, with_i, 1, 56, 56);
  DGEMM_SELECT(618, with_i, 0, 32, 56);
  DGEMM_SELECT(619, with_i, 1, 32, 56);
  DGEMM_SELECT(620, with_i, 0, 32, 56);
  DGEMM_SELECT(621, with_i, 1, 32, 56);
  DGEMM_SELECT(622, with_i, 0, 32, 56);
  DGEMM_SELECT(623, with_i, 1, 32, 56);
  DGEMM_SELECT(624, with_i, 0, 16, 56);
  DGEMM_SELECT(625, with_i, 1, 24, 56);
  DGEMM_SELECT(626, with_i, 0, 24, 56);
  DGEMM_SELECT(627, with_i, 1, 24, 56);
  DGEMM_SELECT(628, with_i, 0, 48, 56);
  DGEMM_SELECT(629, with_i, 1, 24, 56);
  DGEMM_SELECT(630, with_i, 0, 24, 56);
  DGEMM_SELECT(631, with_i, 1, 24, 56);
  DGEMM_SELECT(632, with_i, 0, 48, 56);
  DGEMM_SELECT(633, with_i, 1, 24, 56);
  DGEMM_SELECT(634, with_i, 0, 24, 56);
  DGEMM_SELECT(635, with_i, 1, 24, 56);
  DGEMM_SELECT(636, with_i, 0, 24, 56);
  DGEMM_SELECT(637, with_i, 1, 24, 56);
  DGEMM_SELECT(638, with_i, 0, 24, 56);
  DGEMM_SELECT(639, with_i, 1, 24, 56);
  DGEMM_SELECT(640, with_i, 0, 16, 64);
  DGEMM_SELECT(641, with_i, 1, 32, 64);
  DGEMM_SELECT(642, with_i, 0, 16, 64);
  DGEMM_SELECT(643, with_i, 1, 32, 64);
  DGEMM_SELECT(644, with_i, 0, 32, 64);
  DGEMM_SELECT(645, with_i, 1, 32, 64);
  DGEMM_SELECT(646, with_i, 0, 40, 64);
  DGEMM_SELECT(647, with_i, 1, 32, 64);
  DGEMM_SELECT(648, with_i, 0, 24, 64);
  DGEMM_SELECT(649, with_i, 1, 24, 64);
  DGEMM_SELECT(650, with_i, 0, 24, 64);
  DGEMM_SELECT(651, with_i, 1, 24, 64);
  DGEMM_SELECT(652, with_i, 0, 24, 64);
  DGEMM_SELECT(653, with_i, 1, 24, 64);
  DGEMM_SELECT(654, with_i, 0, 24, 64);
  DGEMM_SELECT(655, with_i, 1, 24, 64);
  DGEMM_SELECT(656, with_i, 0, 16, 64);
  DGEMM_SELECT(657, with_i, 1, 16, 64);
  DGEMM_SELECT(658, with_i, 0, 16, 64);
  DGEMM_SELECT(659, with_i, 1, 16, 64);
  DGEMM_SELECT(660, with_i, 0, 16, 64);
  DGEMM_SELECT(661, with_i, 1, 16, 64);
  DGEMM_SELECT(662, with_i, 0, 16, 64);
  DGEMM_SELECT(663, with_i, 1, 16, 64);
  DGEMM_SELECT(664, with_i, 0, 16, 64);
  DGEMM_SELECT(665, with_i, 1, 16, 64);
  DGEMM_SELECT(666, with_i, 0, 16, 64);
  DGEMM_SELECT(667, with_i, 1, 16, 64);
  DGEMM_SELECT(668, with_i, 0, 16, 64);
  DGEMM_SELECT(669, with_i, 1, 16, 64);
  DGEMM_SELECT(670, with_i, 0, 16, 64);
  DGEMM_SELECT(671, with_i, 1, 16, 64);
  DGEMM_SELECT(672, with_i, 0, 32, 56);
  DGEMM_SELECT(673, with_i, 1, 32, 56);
  DGEMM_SELECT(674, with_i, 0, 32, 56);
  DGEMM_SELECT(675, with_i, 1, 32, 56);
  DGEMM_SELECT(676, with_i, 0, 32, 56);
  DGEMM_SELECT(677, with_i, 1, 32, 56);
  DGEMM_SELECT(678, with_i, 0, 32, 56);
  DGEMM_SELECT(679, with_i, 1, 24, 56);
  DGEMM_SELECT(680, with_i, 0, 40, 56);
  DGEMM_SELECT(681, with_i, 1, 16, 56);
  DGEMM_SELECT(682, with_i, 0, 8, 56);
  DGEMM_SELECT(683, with_i, 1, 16, 56);
  DGEMM_SELECT(684, with_i, 0, 8, 56);
  DGEMM_SELECT(685, with_i, 1, 40, 56);
  DGEMM_SELECT(686, with_i, 0, 40, 56);
  DGEMM_SELECT(687, with_i, 1, 40, 56);
  DGEMM_SELECT(688, with_i, 0, 16, 56);
  DGEMM_SELECT(689, with_i, 1, 16, 56);
  DGEMM_SELECT(690, with_i, 0, 16, 56);
  DGEMM_SELECT(691, with_i, 1, 16, 56);
  DGEMM_SELECT(692, with_i, 0, 16, 56);
  DGEMM_SELECT(693, with_i, 1, 40, 56);
  DGEMM_SELECT(694, with_i, 0, 16, 56);
  DGEMM_SELECT(695, with_i, 1, 16, 56);
  DGEMM_SELECT(696, with_i, 0, 24, 56);
  DGEMM_SELECT(697, with_i, 1, 24, 56);
  DGEMM_SELECT(698, with_i, 0, 24, 56);
  DGEMM_SELECT(699, with_i, 1, 24, 56);
  DGEMM_SELECT(700, with_i, 0, 24, 56);
  DGEMM_SELECT(701, with_i, 1, 24, 56);
  DGEMM_SELECT(702, with_i, 0, 24, 56);
  DGEMM_SELECT(703, with_i, 1, 24, 56);
  DGEMM_SELECT(704, with_i, 0, 32, 64);
  DGEMM_SELECT(705, with_i, 1, 32, 64);
  DGEMM_SELECT(706, with_i, 0, 32, 64);
  DGEMM_SELECT(707, with_i, 1, 32, 64);
  DGEMM_SELECT(708, with_i, 0, 32, 64);
  DGEMM_SELECT(709, with_i, 1, 32, 64);
  DGEMM_SELECT(710, with_i, 0, 32, 64);
  DGEMM_SELECT(711, with_i, 1, 32, 64);
  DGEMM_SELECT(712, with_i, 0, 32, 64);
  DGEMM_SELECT(713, with_i, 1, 32, 64);
  DGEMM_SELECT(714, with_i, 0, 32, 64);
  DGEMM_SELECT(715, with_i, 1, 32, 64);
  DGEMM_SELECT(716, with_i, 0, 32, 64);
  DGEMM_SELECT(717, with_i, 1, 32, 64);
  DGEMM_SELECT(718, with_i, 0, 32, 64);
  DGEMM_SELECT(719, with_i, 1, 32, 64);
  DGEMM_SELECT(720, with_i, 0, 40, 64);
  DGEMM_SELECT(721, with_i, 1, 40, 64);
  DGEMM_SELECT(722, with_i, 0, 40, 64);
  DGEMM_SELECT(723, with_i, 1, 40, 64);
  DGEMM_SELECT(724, with_i, 0, 40, 64);
  DGEMM_SELECT(725, with_i, 1, 40, 64);
  DGEMM_SELECT(726, with_i, 0, 40, 64);
  DGEMM_SELECT(727, with_i, 1, 40, 64);
  DGEMM_SELECT(728, with_i, 0, 56, 56);
  DGEMM_SELECT(729, with_i, 1, 40, 56);
  DGEMM_SELECT(730, with_i, 0, 40, 56);
  DGEMM_SELECT(731, with_i, 1, 40, 56);
  DGEMM_SELECT(732, with_i, 0, 40, 56);
  DGEMM_SELECT(733, with_i, 1, 40, 56);
  DGEMM_SELECT(734, with_i, 0, 40, 56);
  DGEMM_SELECT(735, with_i, 1, 56, 56);
  DGEMM_SELECT(736, with_i, 0, 32, 56);
  DGEMM_SELECT(737, with_i, 1, 32, 56);
  DGEMM_SELECT(738, with_i, 0, 32, 56);
  DGEMM_SELECT(739, with_i, 1, 32, 56);
  DGEMM_SELECT(740, with_i, 0, 32, 56);
  DGEMM_SELECT(741, with_i, 1, 32, 56);
  DGEMM_SELECT(742, with_i, 0, 32, 56);
  DGEMM_SELECT(743, with_i, 1, 32, 56);
  DGEMM_SELECT(744, with_i, 0, 24, 56);
  DGEMM_SELECT(745, with_i, 1, 24, 56);
  DGEMM_SELECT(746, with_i, 0, 24, 56);
  DGEMM_SELECT(747, with_i, 1, 24, 56);
  DGEMM_SELECT(748, with_i, 0, 24, 56);
  DGEMM_SELECT(749, with_i, 1, 24, 56);
  DGEMM_SELECT(750, with_i, 0, 24, 56);
  DGEMM_SELECT(751, with_i, 1, 24, 56);
  DGEMM_SELECT(752, with_i, 0, 16, 56);
  DGEMM_SELECT(753, with_i, 1, 16, 56);
  DGEMM_SELECT(754, with_i, 0, 16, 56);
  DGEMM_SELECT(755, with_i, 1, 16, 56);
  DGEMM_SELECT(756, with_i, 0, 16, 56);
  DGEMM_SELECT(757, with_i, 1, 16, 56);
  DGEMM_SELECT(758, with_i, 0, 16, 56);
  DGEMM_SELECT(759, with_i, 1, 24, 56);
  DGEMM_SELECT(760, with_i, 0, 40, 56);
  DGEMM_SELECT(761, with_i, 1, 40, 56);
  DGEMM_SELECT(762, with_i, 0, 40, 56);
  DGEMM_SELECT(763, with_i, 1, 40, 16);
  DGEMM_SELECT(764, with_i, 0, 40, 56);
  DGEMM_SELECT(765, with_i, 1, 40, 16);
  DGEMM_SELECT(766, with_i, 0, 40, 16);
  DGEMM_SELECT(767, with_i, 1, 40, 56);
  DGEMM_SELECT(768, with_i, 0, 64, 64);
  DGEMM_SELECT(769, with_i, 1, 16, 64);
  DGEMM_SELECT(770, with_i, 0, 16, 64);
  DGEMM_SELECT(771, with_i, 1, 48, 64);
  DGEMM_SELECT(772, with_i, 0, 16, 64);
  DGEMM_SELECT(773, with_i, 1, 32, 64);
  DGEMM_SELECT(774, with_i, 0, 32, 64);
  DGEMM_SELECT(775, with_i, 1, 32, 64);
  DGEMM_SELECT(776, with_i, 0, 32, 64);
  DGEMM_SELECT(777, with_i, 1, 32, 64);
  DGEMM_SELECT(778, with_i, 0, 32, 64);
  DGEMM_SELECT(779, with_i, 1, 32, 64);
  DGEMM_SELECT(780, with_i, 0, 32, 64);
  DGEMM_SELECT(781, with_i, 1, 32, 64);
  DGEMM_SELECT(782, with_i, 0, 32, 64);
  DGEMM_SELECT(783, with_i, 1, 32, 64);
  DGEMM_SELECT(784, with_i, 0, 56, 56);
  DGEMM_SELECT(785, with_i, 1, 56, 56);
  DGEMM_SELECT(786, with_i, 0, 56, 56);
  DGEMM_SELECT(787, with_i, 1, 56, 56);
  DGEMM_SELECT(788, with_i, 0, 56, 56);
  DGEMM_SELECT(789, with_i, 1, 56, 56);
  DGEMM_SELECT(790, with_i, 0, 56, 56);
  DGEMM_SELECT(791, with_i, 1, 56, 56);
  DGEMM_SELECT(792, with_i, 0, 24, 56);
  DGEMM_SELECT(793, with_i, 1, 24, 56);
  DGEMM_SELECT(794, with_i, 0, 24, 56);
  DGEMM_SELECT(795, with_i, 1, 24, 56);
  DGEMM_SELECT(796, with_i, 0, 24, 56);
  DGEMM_SELECT(797, with_i, 1, 24, 56);
  DGEMM_SELECT(798, with_i, 0, 24, 56);
  DGEMM_SELECT(799, with_i, 1, 24, 56);
  DGEMM_SELECT(800, with_i, 0, 40, 56);
  DGEMM_SELECT(801, with_i, 1, 40, 56);
  DGEMM_SELECT(802, with_i, 0, 40, 56);
  DGEMM_SELECT(803, with_i, 1, 40, 56);
  DGEMM_SELECT(804, with_i, 0, 40, 56);
  DGEMM_SELECT(805, with_i, 1, 40, 56);
  DGEMM_SELECT(806, with_i, 0, 40, 56);
  DGEMM_SELECT(807, with_i, 1, 40, 56);
  DGEMM_SELECT(808, with_i, 0, 40, 56);
  DGEMM_SELECT(809, with_i, 1, 40, 56);
  DGEMM_SELECT(810, with_i, 0, 40, 56);
  DGEMM_SELECT(811, with_i, 1, 40, 56);
  DGEMM_SELECT(812, with_i, 0, 40, 56);
  DGEMM_SELECT(813, with_i, 1, 40, 56);
  DGEMM_SELECT(814, with_i, 0, 40, 56);
  DGEMM_SELECT(815, with_i, 1, 40, 56);
  DGEMM_SELECT(816, with_i, 0, 48, 56);
  DGEMM_SELECT(817, with_i, 1, 24, 56);
  DGEMM_SELECT(818, with_i, 0, 24, 56);
  DGEMM_SELECT(819, with_i, 1, 24, 56);
  DGEMM_SELECT(820, with_i, 0, 24, 56);
  DGEMM_SELECT(821, with_i, 1, 48, 56);
  DGEMM_SELECT(822, with_i, 0, 48, 56);
  DGEMM_SELECT(823, with_i, 1, 48, 56);
  DGEMM_SELECT(824, with_i, 0, 48, 56);
  DGEMM_SELECT(825, with_i, 1, 48, 56);
  DGEMM_SELECT(826, with_i, 0, 48, 56);
  DGEMM_SELECT(827, with_i, 1, 48, 56);
  DGEMM_SELECT(828, with_i, 0, 48, 56);
  DGEMM_SELECT(829, with_i, 1, 48, 56);
  DGEMM_SELECT(830, with_i, 0, 48, 56);
  DGEMM_SELECT(831, with_i, 1, 48, 56);
  DGEMM_SELECT(832, with_i, 0, 32, 64);
  DGEMM_SELECT(833, with_i, 1, 32, 64);
  DGEMM_SELECT(834, with_i, 0, 32, 64);
  DGEMM_SELECT(835, with_i, 1, 32, 64);
  DGEMM_SELECT(836, with_i, 0, 32, 64);
  DGEMM_SELECT(837, with_i, 1, 32, 64);
  DGEMM_SELECT(838, with_i, 0, 32, 64);
  DGEMM_SELECT(839, with_i, 1, 32, 64);
  DGEMM_SELECT(840, with_i, 0, 40, 56);
  DGEMM_SELECT(841, with_i, 1, 40, 56);
  DGEMM_SELECT(842, with_i, 0, 40, 56);
  DGEMM_SELECT(843, with_i, 1, 40, 56);
  DGEMM_SELECT(844, with_i, 0, 40, 56);
  DGEMM_SELECT(845, with_i, 1, 40, 56);
  DGEMM_SELECT(846, with_i, 0, 40, 56);
  DGEMM_SELECT(847, with_i, 1, 40, 56);
  DGEMM_SELECT(848, with_i, 0, 40, 56);
  DGEMM_SELECT(849, with_i, 1, 40, 56);
  DGEMM_SELECT(850, with_i, 0, 40, 56);
  DGEMM_SELECT(851, with_i, 1, 16, 56);
  DGEMM_SELECT(852, with_i, 0, 16, 56);
  DGEMM_SELECT(853, with_i, 1, 16, 56);
  DGEMM_SELECT(854, with_i, 0, 16, 56);
  DGEMM_SELECT(855, with_i, 1, 56, 56);
  DGEMM_SELECT(856, with_i, 0, 40, 56);
  DGEMM_SELECT(857, with_i, 1, 40, 56);
  DGEMM_SELECT(858, with_i, 0, 40, 56);
  DGEMM_SELECT(859, with_i, 1, 40, 56);
  DGEMM_SELECT(860, with_i, 0, 40, 56);
  DGEMM_SELECT(861, with_i, 1, 40, 56);
  DGEMM_SELECT(862, with_i, 0, 40, 56);
  DGEMM_SELECT(863, with_i, 1, 40, 56);
  DGEMM_SELECT(864, with_i, 0, 48, 56);
  DGEMM_SELECT(865, with_i, 1, 48, 56);
  DGEMM_SELECT(866, with_i, 0, 48, 56);
  DGEMM_SELECT(867, with_i, 1, 48, 56);
  DGEMM_SELECT(868, with_i, 0, 48, 56);
  DGEMM_SELECT(869, with_i, 1, 32, 56);
  DGEMM_SELECT(870, with_i, 0, 32, 56);
  DGEMM_SELECT(871, with_i, 1, 48, 56);
  DGEMM_SELECT(872, with_i, 0, 48, 56);
  DGEMM_SELECT(873, with_i, 1, 48, 56);
  DGEMM_SELECT(874, with_i, 0, 48, 56);
  DGEMM_SELECT(875, with_i, 1, 48, 56);
  DGEMM_SELECT(876, with_i, 0, 48, 56);
  DGEMM_SELECT(877, with_i, 1, 48, 56);
  DGEMM_SELECT(878, with_i, 0, 48, 56);
  DGEMM_SELECT(879, with_i, 1, 48, 56);
  DGEMM_SELECT(880, with_i, 0, 40, 56);
  DGEMM_SELECT(881, with_i, 1, 40, 56);
  DGEMM_SELECT(882, with_i, 0, 40, 56);
  DGEMM_SELECT(883, with_i, 1, 40, 56);
  DGEMM_SELECT(884, with_i, 0, 40, 56);
  DGEMM_SELECT(885, with_i, 1, 40, 56);
  DGEMM_SELECT(886, with_i, 0, 40, 56);
  DGEMM_SELECT(887, with_i, 1, 40, 56);
  DGEMM_SELECT(888, with_i, 0, 40, 56);
  DGEMM_SELECT(889, with_i, 1, 40, 56);
  DGEMM_SELECT(890, with_i, 0, 40, 56);
  DGEMM_SELECT(891, with_i, 1, 40, 56);
  DGEMM_SELECT(892, with_i, 0, 40, 56);
  DGEMM_SELECT(893, with_i, 1, 40, 16);
  DGEMM_SELECT(894, with_i, 0, 24, 56);
  DGEMM_SELECT(895, with_i, 1, 24, 56);
  DGEMM_SELECT(896, with_i, 0, 24, 64);
  DGEMM_SELECT(897, with_i, 1, 32, 64);
  DGEMM_SELECT(898, with_i, 0, 32, 56);
  DGEMM_SELECT(899, with_i, 1, 32, 64);
  DGEMM_SELECT(900, with_i, 0, 32, 64);
  DGEMM_SELECT(901, with_i, 1, 32, 64);
  DGEMM_SELECT(902, with_i, 0, 32, 64);
  DGEMM_SELECT(903, with_i, 1, 32, 64);
  DGEMM_SELECT(904, with_i, 0, 32, 64);
  DGEMM_SELECT(905, with_i, 1, 32, 64);
  DGEMM_SELECT(906, with_i, 0, 32, 64);
  DGEMM_SELECT(907, with_i, 1, 32, 64);
  DGEMM_SELECT(908, with_i, 0, 32, 64);
  DGEMM_SELECT(909, with_i, 1, 32, 64);
  DGEMM_SELECT(910, with_i, 0, 32, 64);
  DGEMM_SELECT(911, with_i, 1, 32, 64);
  DGEMM_SELECT(912, with_i, 0, 48, 56);
  DGEMM_SELECT(913, with_i, 1, 48, 56);
  DGEMM_SELECT(914, with_i, 0, 48, 56);
  DGEMM_SELECT(915, with_i, 1, 48, 56);
  DGEMM_SELECT(916, with_i, 0, 48, 56);
  DGEMM_SELECT(917, with_i, 1, 48, 56);
  DGEMM_SELECT(918, with_i, 0, 48, 56);
  DGEMM_SELECT(919, with_i, 1, 48, 56);
  DGEMM_SELECT(920, with_i, 0, 40, 56);
  DGEMM_SELECT(921, with_i, 1, 40, 64);
  DGEMM_SELECT(922, with_i, 0, 40, 64);
  DGEMM_SELECT(923, with_i, 1, 40, 64);
  DGEMM_SELECT(924, with_i, 0, 40, 64);
  DGEMM_SELECT(925, with_i, 1, 40, 64);
  DGEMM_SELECT(926, with_i, 0, 40, 64);
  DGEMM_SELECT(927, with_i, 1, 40, 56);
  DGEMM_SELECT(928, with_i, 0, 32, 64);
  DGEMM_SELECT(929, with_i, 1, 32, 64);
  DGEMM_SELECT(930, with_i, 0, 32, 64);
  DGEMM_SELECT(931, with_i, 1, 32, 64);
  DGEMM_SELECT(932, with_i, 0, 32, 64);
  DGEMM_SELECT(933, with_i, 1, 32, 64);
  DGEMM_SELECT(934, with_i, 0, 32, 64);
  DGEMM_SELECT(935, with_i, 1, 32, 64);
  DGEMM_SELECT(936, with_i, 0, 24, 64);
  DGEMM_SELECT(937, with_i, 1, 32, 64);
  DGEMM_SELECT(938, with_i, 0, 24, 64);
  DGEMM_SELECT(939, with_i, 1, 32, 64);
  DGEMM_SELECT(940, with_i, 0, 24, 64);
  DGEMM_SELECT(941, with_i, 1, 32, 64);
  DGEMM_SELECT(942, with_i, 0, 24, 64);
  DGEMM_SELECT(943, with_i, 1, 32, 64);
  DGEMM_SELECT(944, with_i, 0, 32, 64);
  DGEMM_SELECT(945, with_i, 1, 32, 64);
  DGEMM_SELECT(946, with_i, 0, 32, 64);
  DGEMM_SELECT(947, with_i, 1, 32, 64);
  DGEMM_SELECT(948, with_i, 0, 32, 64);
  DGEMM_SELECT(949, with_i, 1, 32, 64);
  DGEMM_SELECT(950, with_i, 0, 24, 64);
  DGEMM_SELECT(951, with_i, 1, 32, 64);
  DGEMM_SELECT(952, with_i, 0, 56, 56);
  DGEMM_SELECT(953, with_i, 1, 56, 56);
  DGEMM_SELECT(954, with_i, 0, 56, 56);
  DGEMM_SELECT(955, with_i, 1, 56, 56);
  DGEMM_SELECT(956, with_i, 0, 56, 56);
  DGEMM_SELECT(957, with_i, 1, 56, 56);
  DGEMM_SELECT(958, with_i, 0, 56, 56);
  DGEMM_SELECT(959, with_i, 1, 56, 56);
  DGEMM_SELECT(960, with_i, 0, 48, 56);
  DGEMM_SELECT(961, with_i, 1, 40, 64);
  DGEMM_SELECT(962, with_i, 0, 40, 64);
  DGEMM_SELECT(963, with_i, 1, 40, 64);
  DGEMM_SELECT(964, with_i, 0, 40, 64);
  DGEMM_SELECT(965, with_i, 1, 40, 64);
  DGEMM_SELECT(966, with_i, 0, 40, 64);
  DGEMM_SELECT(967, with_i, 1, 40, 64);
  DGEMM_SELECT(968, with_i, 0, 40, 64);
  DGEMM_SELECT(969, with_i, 1, 40, 64);
  DGEMM_SELECT(970, with_i, 0, 40, 64);
  DGEMM_SELECT(971, with_i, 1, 40, 64);
  DGEMM_SELECT(972, with_i, 0, 40, 64);
  DGEMM_SELECT(973, with_i, 1, 40, 64);
  DGEMM_SELECT(974, with_i, 0, 40, 64);
  DGEMM_SELECT(975, with_i, 1, 40, 64);
  DGEMM_SELECT(976, with_i, 0, 40, 64);
  DGEMM_SELECT(977, with_i, 1, 32, 64);
  DGEMM_SELECT(978, with_i, 0, 40, 64);
  DGEMM_SELECT(979, with_i, 1, 40, 64);
  DGEMM_SELECT(980, with_i, 0, 40, 64);
  DGEMM_SELECT(981, with_i, 1, 40, 64);
  DGEMM_SELECT(982, with_i, 0, 40, 64);
  DGEMM_SELECT(983, with_i, 1, 40, 64);
  DGEMM_SELECT(984, with_i, 0, 24, 64);
  DGEMM_SELECT(985, with_i, 1, 24, 64);
  DGEMM_SELECT(986, with_i, 0, 24, 64);
  DGEMM_SELECT(987, with_i, 1, 24, 64);
  DGEMM_SELECT(988, with_i, 0, 24, 64);
  DGEMM_SELECT(989, with_i, 1, 24, 64);
  DGEMM_SELECT(990, with_i, 0, 24, 64);
  DGEMM_SELECT(991, with_i, 1, 24, 64);
  DGEMM_SELECT(992, with_i, 0, 32, 64);
  DGEMM_SELECT(993, with_i, 1, 32, 64);
  DGEMM_SELECT(994, with_i, 0, 32, 64);
  DGEMM_SELECT(995, with_i, 1, 32, 64);
  DGEMM_SELECT(996, with_i, 0, 32, 64);
  DGEMM_SELECT(997, with_i, 1, 32, 64);
  DGEMM_SELECT(998, with_i, 0, 32, 64);
  DGEMM_SELECT(999, with_i, 1, 32, 64);
  DGEMM_SELECT(1000, with_i, 0, 40, 64);
  DGEMM_SELECT(1001, with_i, 1, 40, 64);
  DGEMM_SELECT(1002, with_i, 0, 40, 64);
  DGEMM_SELECT(1003, with_i, 1, 40, 64);
  DGEMM_SELECT(1004, with_i, 0, 40, 64);
  DGEMM_SELECT(1005, with_i, 1, 40, 56);
  DGEMM_SELECT(1006, with_i, 0, 40, 56);
  DGEMM_SELECT(1007, with_i, 1, 40, 64);
  DGEMM_SELECT(1008, with_i, 0, 48, 56);
  DGEMM_SELECT(1009, with_i, 1, 40, 56);
  DGEMM_SELECT(1010, with_i, 0, 40, 56);
  DGEMM_SELECT(1011, with_i, 1, 48, 56);
  DGEMM_SELECT(1012, with_i, 0, 48, 56);
  DGEMM_SELECT(1013, with_i, 1, 48, 56);
  DGEMM_SELECT(1014, with_i, 0, 48, 56);
  DGEMM_SELECT(1015, with_i, 1, 56, 56);
  DGEMM_SELECT(1016, with_i, 0, 56, 56);
  DGEMM_SELECT(1017, with_i, 1, 56, 16);
  DGEMM_SELECT(1018, with_i, 0, 56, 56);
  DGEMM_SELECT(1019, with_i, 1, 56, 56);
  DGEMM_SELECT(1020, with_i, 0, 48, 56);
  DGEMM_SELECT(1021, with_i, 1, 56, 56);
  DGEMM_SELECT(1022, with_i, 0, 56, 56);
  DGEMM_SELECT(1023, with_i, 1, 56, 56);
  DGEMM_SELECT(1024, with_i, 0, 64, 64);
  DGEMM_SELECT(1025, with_i, 1, 64, 64);
  DGEMM_SELECT(1026, with_i, 0, 64, 64);
  DGEMM_SELECT(1027, with_i, 1, 64, 64);
  DGEMM_SELECT(1028, with_i, 0, 64, 64);
  DGEMM_SELECT(1029, with_i, 1, 64, 64);
  DGEMM_SELECT(1030, with_i, 0, 64, 64);
  DGEMM_SELECT(1031, with_i, 1, 64, 56);
  DGEMM_SELECT(1032, with_i, 0, 64, 56);
  DGEMM_SELECT(1033, with_i, 1, 32, 64);
  DGEMM_SELECT(1034, with_i, 0, 32, 64);
  DGEMM_SELECT(1035, with_i, 1, 32, 64);
  DGEMM_SELECT(1036, with_i, 0, 32, 64);
  DGEMM_SELECT(1037, with_i, 1, 32, 64);
  DGEMM_SELECT(1038, with_i, 0, 32, 64);
  DGEMM_SELECT(1039, with_i, 1, 32, 64);
  DGEMM_SELECT(1040, with_i, 0, 40, 64);
  DGEMM_SELECT(1041, with_i, 1, 40, 64);
  DGEMM_SELECT(1042, with_i, 0, 40, 64);
  DGEMM_SELECT(1043, with_i, 1, 40, 64);
  DGEMM_SELECT(1044, with_i, 0, 40, 64);
  DGEMM_SELECT(1045, with_i, 1, 40, 64);
  DGEMM_SELECT(1046, with_i, 0, 40, 64);
  DGEMM_SELECT(1047, with_i, 1, 40, 64);
  DGEMM_SELECT(1048, with_i, 0, 40, 64);
  DGEMM_SELECT(1049, with_i, 1, 40, 64);
  DGEMM_SELECT(1050, with_i, 0, 40, 64);
  DGEMM_SELECT(1051, with_i, 1, 40, 64);
  DGEMM_SELECT(1052, with_i, 0, 40, 64);
  DGEMM_SELECT(1053, with_i, 1, 40, 64);
  DGEMM_SELECT(1054, with_i, 0, 40, 64);
  DGEMM_SELECT(1055, with_i, 1, 40, 64);
  DGEMM_SELECT(1056, with_i, 0, 32, 64);
  DGEMM_SELECT(1057, with_i, 1, 32, 64);
  DGEMM_SELECT(1058, with_i, 0, 32, 64);
  DGEMM_SELECT(1059, with_i, 1, 32, 64);
  DGEMM_SELECT(1060, with_i, 0, 32, 64);
  DGEMM_SELECT(1061, with_i, 1, 32, 64);
  DGEMM_SELECT(1062, with_i, 0, 32, 64);
  DGEMM_SELECT(1063, with_i, 1, 32, 64);
  DGEMM_SELECT(1064, with_i, 0, 56, 56);
  DGEMM_SELECT(1065, with_i, 1, 48, 56);
  DGEMM_SELECT(1066, with_i, 0, 48, 56);
  DGEMM_SELECT(1067, with_i, 1, 56, 56);
  DGEMM_SELECT(1068, with_i, 0, 56, 56);
  DGEMM_SELECT(1069, with_i, 1, 56, 56);
  DGEMM_SELECT(1070, with_i, 0, 56, 56);
  DGEMM_SELECT(1071, with_i, 1, 56, 56);
  DGEMM_SELECT(1072, with_i, 0, 48, 56);
  DGEMM_SELECT(1073, with_i, 1, 48, 56);
  DGEMM_SELECT(1074, with_i, 0, 48, 56);
  DGEMM_SELECT(1075, with_i, 1, 56, 56);
  DGEMM_SELECT(1076, with_i, 0, 48, 56);
  DGEMM_SELECT(1077, with_i, 1, 48, 56);
  DGEMM_SELECT(1078, with_i, 0, 48, 56);
  DGEMM_SELECT(1079, with_i, 1, 56, 56);
  DGEMM_SELECT(1080, with_i, 0, 40, 56);
  DGEMM_SELECT(1081, with_i, 1, 40, 56);
  DGEMM_SELECT(1082, with_i, 0, 40, 56);
  DGEMM_SELECT(1083, with_i, 1, 40, 56);
  DGEMM_SELECT(1084, with_i, 0, 40, 56);
  DGEMM_SELECT(1085, with_i, 1, 40, 56);
  DGEMM_SELECT(1086, with_i, 0, 40, 56);
  DGEMM_SELECT(1087, with_i, 1, 40, 56);
  DGEMM_SELECT(1088, with_i, 0, 32, 64);
  DGEMM_SELECT(1089, with_i, 1, 32, 64);
  DGEMM_SELECT(1090, with_i, 0, 32, 64);
  DGEMM_SELECT(1091, with_i, 1, 32, 64);
  DGEMM_SELECT(1092, with_i, 0, 32, 64);
  DGEMM_SELECT(1093, with_i, 1, 32, 64);
  DGEMM_SELECT(1094, with_i, 0, 32, 64);
  DGEMM_SELECT(1095, with_i, 1, 32, 64);
  DGEMM_SELECT(1096, with_i, 0, 32, 64);
  DGEMM_SELECT(1097, with_i, 1, 32, 64);
  DGEMM_SELECT(1098, with_i, 0, 32, 64);
  DGEMM_SELECT(1099, with_i, 1, 32, 64);
  DGEMM_SELECT(1100, with_i, 0, 32, 64);
  DGEMM_SELECT(1101, with_i, 1, 32, 64);
  DGEMM_SELECT(1102, with_i, 0, 32, 64);
  DGEMM_SELECT(1103, with_i, 1, 32, 64);
  DGEMM_SELECT(1104, with_i, 0, 48, 64);
  DGEMM_SELECT(1105, with_i, 1, 48, 64);
  DGEMM_SELECT(1106, with_i, 0, 48, 64);
  DGEMM_SELECT(1107, with_i, 1, 48, 64);
  DGEMM_SELECT(1108, with_i, 0, 48, 64);
  DGEMM_SELECT(1109, with_i, 1, 48, 64);
  DGEMM_SELECT(1110, with_i, 0, 48, 64);
  DGEMM_SELECT(1111, with_i, 1, 48, 64);
  DGEMM_SELECT(1112, with_i, 0, 48, 64);
  DGEMM_SELECT(1113, with_i, 1, 48, 64);
  DGEMM_SELECT(1114, with_i, 0, 48, 64);
  DGEMM_SELECT(1115, with_i, 1, 48, 64);
  DGEMM_SELECT(1116, with_i, 0, 48, 64);
  DGEMM_SELECT(1117, with_i, 1, 48, 64);
  DGEMM_SELECT(1118, with_i, 0, 48, 64);
  DGEMM_SELECT(1119, with_i, 1, 48, 64);
  DGEMM_SELECT(1120, with_i, 0, 40, 56);
  DGEMM_SELECT(1121, with_i, 1, 40, 56);
  DGEMM_SELECT(1122, with_i, 0, 40, 56);
  DGEMM_SELECT(1123, with_i, 1, 40, 56);
  DGEMM_SELECT(1124, with_i, 0, 40, 56);
  DGEMM_SELECT(1125, with_i, 1, 32, 56);
  DGEMM_SELECT(1126, with_i, 0, 32, 56);
  DGEMM_SELECT(1127, with_i, 1, 40, 56);
  DGEMM_SELECT(1128, with_i, 0, 40, 56);
  DGEMM_SELECT(1129, with_i, 1, 40, 56);
  DGEMM_SELECT(1130, with_i, 0, 40, 56);
  DGEMM_SELECT(1131, with_i, 1, 40, 56);
  DGEMM_SELECT(1132, with_i, 0, 40, 56);
  DGEMM_SELECT(1133, with_i, 1, 40, 56);
  DGEMM_SELECT(1134, with_i, 0, 40, 56);
  DGEMM_SELECT(1135, with_i, 1, 40, 56);
  DGEMM_SELECT(1136, with_i, 0, 40, 56);
  DGEMM_SELECT(1137, with_i, 1, 40, 56);
  DGEMM_SELECT(1138, with_i, 0, 32, 56);
  DGEMM_SELECT(1139, with_i, 1, 40, 56);
  DGEMM_SELECT(1140, with_i, 0, 40, 56);
  DGEMM_SELECT(1141, with_i, 1, 40, 56);
  DGEMM_SELECT(1142, with_i, 0, 40, 56);
  DGEMM_SELECT(1143, with_i, 1, 40, 56);
  DGEMM_SELECT(1144, with_i, 0, 40, 56);
  DGEMM_SELECT(1145, with_i, 1, 40, 56);
  DGEMM_SELECT(1146, with_i, 0, 40, 56);
  DGEMM_SELECT(1147, with_i, 1, 40, 56);
  DGEMM_SELECT(1148, with_i, 0, 40, 56);
  DGEMM_SELECT(1149, with_i, 1, 24, 56);
  DGEMM_SELECT(1150, with_i, 0, 24, 56);
  DGEMM_SELECT(1151, with_i, 1, 24, 56);
  DGEMM_SELECT(1152, with_i, 0, 24, 64);
  DGEMM_SELECT(1153, with_i, 1, 32, 64);
  DGEMM_SELECT(1154, with_i, 0, 32, 64);
  DGEMM_SELECT(1155, with_i, 1, 32, 64);
  DGEMM_SELECT(1156, with_i, 0, 32, 64);
  DGEMM_SELECT(1157, with_i, 1, 32, 64);
  DGEMM_SELECT(1158, with_i, 0, 32, 64);
  DGEMM_SELECT(1159, with_i, 1, 32, 64);
  DGEMM_SELECT(1160, with_i, 0, 40, 64);
  DGEMM_SELECT(1161, with_i, 1, 40, 64);
  DGEMM_SELECT(1162, with_i, 0, 40, 64);
  DGEMM_SELECT(1163, with_i, 1, 40, 64);
  DGEMM_SELECT(1164, with_i, 0, 40, 64);
  DGEMM_SELECT(1165, with_i, 1, 40, 64);
  DGEMM_SELECT(1166, with_i, 0, 40, 64);
  DGEMM_SELECT(1167, with_i, 1, 40, 64);
  DGEMM_SELECT(1168, with_i, 0, 40, 64);
  DGEMM_SELECT(1169, with_i, 1, 40, 64);
  DGEMM_SELECT(1170, with_i, 0, 40, 64);
  DGEMM_SELECT(1171, with_i, 1, 40, 64);
  DGEMM_SELECT(1172, with_i, 0, 40, 64);
  DGEMM_SELECT(1173, with_i, 1, 40, 64);
  DGEMM_SELECT(1174, with_i, 0, 40, 64);
  DGEMM_SELECT(1175, with_i, 1, 40, 64);
  DGEMM_SELECT(1176, with_i, 0, 56, 56);
  DGEMM_SELECT(1177, with_i, 1, 56, 56);
  DGEMM_SELECT(1178, with_i, 0, 56, 56);
  DGEMM_SELECT(1179, with_i, 1, 56, 56);
  DGEMM_SELECT(1180, with_i, 0, 56, 56);
  DGEMM_SELECT(1181, with_i, 1, 56, 56);
  DGEMM_SELECT(1182, with_i, 0, 56, 56);
  DGEMM_SELECT(1183, with_i, 1, 56, 56);
  DGEMM_SELECT(1184, with_i, 0, 32, 56);
  DGEMM_SELECT(1185, with_i, 1, 32, 56);
  DGEMM_SELECT(1186, with_i, 0, 32, 56);
  DGEMM_SELECT(1187, with_i, 1, 32, 56);
  DGEMM_SELECT(1188, with_i, 0, 32, 56);
  DGEMM_SELECT(1189, with_i, 1, 32, 56);
  DGEMM_SELECT(1190, with_i, 0, 32, 56);
  DGEMM_SELECT(1191, with_i, 1, 32, 56);
  DGEMM_SELECT(1192, with_i, 0, 32, 56);
  DGEMM_SELECT(1193, with_i, 1, 24, 56);
  DGEMM_SELECT(1194, with_i, 0, 24, 56);
  DGEMM_SELECT(1195, with_i, 1, 32, 56);
  DGEMM_SELECT(1196, with_i, 0, 32, 56);
  DGEMM_SELECT(1197, with_i, 1, 32, 56);
  DGEMM_SELECT(1198, with_i, 0, 32, 56);
  DGEMM_SELECT(1199, with_i, 1, 32, 56);
	default:
	  if (lda % 2 == 0) {
	    transpose_B_0_32_32(lda, B);
	    SQUARE_DGEMM(with_i, 0, 8, 8, 64)(lda, A, B_global_transpose, C);
	  }
	  else {
	    transpose_B_0_32_32(lda, B);
	    SQUARE_DGEMM(with_i, 1, 8, 8, 64)(lda, A, B_global_transpose, C);
		} 
	}
}

