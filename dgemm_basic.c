#include <immintrin.h>

const char* dgemm_desc = "Basic, three-loop dgemm.";

#ifndef L1_SIZE
#define L1_SIZE ((int) 40)
#endif

#ifndef L2_SIZE
#define L2_SIZE ((int) 280)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/

void copy_aligned_block(const int lda, const int AllOC_SIZE, const int M, const int N, const double *src, double *dst)
{
    int i, j;

    // Copy block row by row, but use AllOC_SIZE for the dst stride
    for (j = 0; j < N; ++j) {
        for (i = 0; i < M; ++i) {
            dst[j * AllOC_SIZE + i] = src[j * lda + i]; // Copy each element from src to aligned buffer
        }

        for (i = M; i < AllOC_SIZE; ++i) {
             dst[j * AllOC_SIZE + i] = 0.0;            // Zero padding for the remaining row
        }
    }

    // Zero padding for the remaining columns
    for (j = N; j < AllOC_SIZE; ++j) {
        for (i = 0; i < AllOC_SIZE; ++i) {
            dst[j * AllOC_SIZE + i] = 0.0;
        }
    }


}



void basic_dgemm(const int lda, const int AllOC_SIZE, const int M, const int N, const int K, const int MM, const int NN, const int KK,
                 const double *A, const double *B, double *C)
{
    int i, j, k;

    for (j = 0; j < NN; ++j) {

        for (k = 0; k < KK; ++k) {
            
            double bjk = B[j * AllOC_SIZE + k];

            for (i = 0; i < MM; ++i) {
                
                C[j * lda + i] += A[k * AllOC_SIZE + i] * bjk;
            }

        }
    }
}

void do_block(const int lda, const int AllOC_SIZE, const int M, const int N, const int K,
              const double *A, const double *B, double *C,
              const int l1_i, const int l1_j, const int l1_k)
{
    const int MM = (l1_i + L1_SIZE > M? M - l1_i : L1_SIZE);

    const int NN = (l1_j + L1_SIZE > N? N - l1_j : L1_SIZE);

    const int KK = (l1_k + L1_SIZE > K? K - l1_k : L1_SIZE);

    basic_dgemm(lda, AllOC_SIZE, M, N, K, MM, NN, KK,
                A + l1_i + l1_k * AllOC_SIZE, B + l1_k + l1_j * AllOC_SIZE, C + l1_i + l1_j * lda);

}

void multi_level_block(const int lda, const int AllOC_SIZE, const int M, const int N, const int K, 
                        const double *aligned_A, const double *aligned_B, double *C, 
                        const int l2_i, const int l2_j, const int l2_k) 
{

    int i, j, k;

    for (j = 0; j < N; j += L1_SIZE) {

        for (i = 0; i < M; i += L1_SIZE) {

            for (k = 0; k < K; k += L1_SIZE) {

                do_block(lda, AllOC_SIZE, M, N, K, aligned_A, aligned_B, C + l2_i + l2_j * lda, i, j, k);

            }

        }

    }

}

void square_dgemm(const int lda, const double *A, const double *B, double *C)
{   
    int i, j, k;

    int AllOC_SIZE = L2_SIZE;

    if(lda < L2_SIZE) {
        AllOC_SIZE = lda % 8? ((lda / 8 + 1) * 8): lda;
    }

    double* aligned_A = (double*) _mm_malloc(AllOC_SIZE * AllOC_SIZE * sizeof(double), 64);
    double* aligned_B = (double*) _mm_malloc(AllOC_SIZE * AllOC_SIZE * sizeof(double), 64);

    for (j = 0; j < lda; j += L2_SIZE) {

        const int N = (j + L2_SIZE > lda? lda - j : L2_SIZE);

        for (i = 0; i < lda; i += L2_SIZE) {

            const int M = (i + L2_SIZE > lda? lda - i : L2_SIZE);

            for (k = 0; k < lda; k += L2_SIZE) {

                const int K = (k + L2_SIZE > lda? lda - k : L2_SIZE);

                copy_aligned_block(lda, AllOC_SIZE, M, K, A + i + k * lda, aligned_A);
                copy_aligned_block(lda, AllOC_SIZE, K, N, B + k + j * lda, aligned_B);

                multi_level_block(lda, AllOC_SIZE, M, N, K, aligned_A, aligned_B, C, i, j, k);

            }

        }

    }

    _mm_free(aligned_A);
    _mm_free(aligned_B);

}

