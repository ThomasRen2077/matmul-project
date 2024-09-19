const char* dgemm_desc = "My awesome dgemm.";

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

void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double *A, const double *B, double *C)
{
    int i, j, k;

    for (j = 0; j < N; ++j) {

        for (k = 0; k < K; ++k) {
            
            double bjk = B[j * lda + k];

            for (i = 0; i < M; ++i) {
                
                C[j * lda + i] += A[k * lda + i] * bjk;
            }

        }
    }
}

void do_block(const int lda, const int M, const int N, const int K,
              const double *A, const double *B, double *C,
              const int l1_i, const int l1_j, const int l1_k)
{
    const int MM = (l1_i + L1_SIZE > M? M - l1_i : L1_SIZE);

    const int NN = (l1_j + L1_SIZE > N? N - l1_j : L1_SIZE);

    const int KK = (l1_k + L1_SIZE > K? K - l1_k : L1_SIZE);

    basic_dgemm(lda, MM, NN, KK,
                A + l1_i + l1_k * lda, B + l1_k + l1_j * lda, C + l1_i + l1_j * lda);

}

void multi_level_block(const int lda, const double *A, const double *B, double *C, const int l2_i, const int l2_j, const int l2_k) {

    const int M = (l2_i + L2_SIZE > lda? lda - l2_i : L2_SIZE);
    const int N = (l2_j + L2_SIZE > lda? lda - l2_j : L2_SIZE);
    const int K = (l2_k + L2_SIZE > lda? lda - l2_k : L2_SIZE);

    int i, j, k;

    for (j = 0; j < N; j += L1_SIZE) {

        for (i = 0; i < M; i += L1_SIZE) {

            for (k = 0; k < K; k += L1_SIZE) {

                do_block(lda, M, N, K, A + l2_i + l2_k * lda, B + l2_k + l2_j * lda, C + l2_i + l2_j * lda, i, j, k);

            }

        }

    }

}

void square_dgemm(const int lda, const double *A, const double *B, double *C)
{   
    int i, j, k;

    for (j = 0; j < lda; j += L2_SIZE) {

        for (i = 0; i < lda; i += L2_SIZE) {

            for (k = 0; k < lda; k += L2_SIZE) {

                multi_level_block(lda, A, B, C, i, j, k);

            }

        }

    }

}

