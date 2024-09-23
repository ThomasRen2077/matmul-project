#include <immintrin.h>
const char* dgemm_desc = "My awesome dgemm.";

#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define L1_SIZE 40
#define L2_SIZE 280

double *newA;
double *newB;

static void init_maps(int size) {
    newA = _mm_malloc(sizeof(double) * size * size, 64);
    newB = _mm_malloc(sizeof(double) * size * size, 64);
}

static void free_maps() {
    free(newA);
    free(newB);
}

static void micro_kernal(int lda, int K, double *a, double *b, double *c) {
    __m512d a_vec, c1_vec, c2_vec, c3_vec, c4_vec;
    __m512d c5_vec, c6_vec, c7_vec, c8_vec;

    double *c2 = c + lda;
    double *c3 = c2 + lda;
    double *c4 = c3 + lda;
    double *c5 = c4 + lda;
    double *c6 = c5 + lda;
    double *c7 = c6 + lda;
    double *c8 = c7 + lda;


    c1_vec = _mm512_loadu_pd(c);
    c2_vec = _mm512_loadu_pd(c2);
    c3_vec = _mm512_loadu_pd(c3);
    c4_vec = _mm512_loadu_pd(c4);
    c5_vec = _mm512_loadu_pd(c5);
    c6_vec = _mm512_loadu_pd(c6);
    c7_vec = _mm512_loadu_pd(c7);
    c8_vec = _mm512_loadu_pd(c8);


    while(K > 0){
        a_vec = _mm512_load_pd(a);
        a += 8;

        c1_vec = _mm512_fmadd_pd(a_vec, _mm512_set1_pd(*(b++)), c1_vec);
        c2_vec = _mm512_fmadd_pd(a_vec, _mm512_set1_pd(*(b++)), c2_vec);
        c3_vec = _mm512_fmadd_pd(a_vec, _mm512_set1_pd(*(b++)), c3_vec);
        c4_vec = _mm512_fmadd_pd(a_vec, _mm512_set1_pd(*(b++)), c4_vec);
        c5_vec = _mm512_fmadd_pd(a_vec, _mm512_set1_pd(*(b++)), c5_vec);
        c6_vec = _mm512_fmadd_pd(a_vec, _mm512_set1_pd(*(b++)), c6_vec);
        c7_vec = _mm512_fmadd_pd(a_vec, _mm512_set1_pd(*(b++)), c7_vec);
        c8_vec = _mm512_fmadd_pd(a_vec, _mm512_set1_pd(*(b++)), c8_vec);

        --K;
    }

    _mm512_storeu_pd(c, c1_vec);
    _mm512_storeu_pd(c2, c2_vec);
    _mm512_storeu_pd(c3, c3_vec);
    _mm512_storeu_pd(c4, c4_vec);
    _mm512_storeu_pd(c5, c5_vec);
    _mm512_storeu_pd(c6, c6_vec);
    _mm512_storeu_pd(c7, c7_vec);
    _mm512_storeu_pd(c8, c8_vec);

}

static void copy(int lda, int K, double *from, double *to){
    for (int i = 0; i < K; ++i) {
        to[0] = from[0];
        to[1] = from[1];
        to[2] = from[2];
        to[3] = from[3];
        to[4] = from[4];
        to[5] = from[5];
        to[6] = from[6];
        to[7] = from[7];
        from += lda;
        to += 8;
    }
}

static void copy_remainder(int lda, int K, double* from, double *to, int remainder){
    for (int i = 0; i < K; ++i) {
        int j = 0;
        while(j < remainder){
            to[j] = from[j];
            ++j;
        }
        while(j < 8){
            to[j] = 0;
            ++j;
        }
        from += lda;
        to += 8;
    }
}

static void transpose(int lda, int K, double *from, double *to) {
    double *p1, *p2, *p3, *p4;
    double *p5, *p6, *p7, *p8;

    p1 = from;
    p2 = p1 + lda;
    p3 = p2 + lda;
    p4 = p3 + lda;
    p5 = p4 + lda;
    p6 = p5 + lda;
    p7 = p6 + lda;
    p8 = p7 + lda;

    for (int i = 0; i < K; ++i) {
        to[0] = *p1++;
        to[1] = *p2++;
        to[2] = *p3++;
        to[3] = *p4++;
        to[4] = *p5++;
        to[5] = *p6++;
        to[6] = *p7++;
        to[7] = *p8++;
        to += 8;
    }
}

static void transpose_remainder(int lda, int K, double* from, double *to, int remainder){
    for (int i = 0; i < K; ++i) {
        int j = 0;
        while(j < remainder){
            to[j] = from[j*lda];
            ++j;
        }
        while(j < 8){
            to[j] = 0;
            ++j;
        }
        to+=8;
        ++from;
    }
}

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */

static void do_block(int lda, int M, int N, int K, double *A, double *B, double *C) {

    int x = M - M % 8;
    for (int i = 0; i < x; i += 8){
        copy(lda, K, A + i, newA + K * i); //copy A to make sure they are aligned
    }

    if  (M % 8 > 0){
        copy_remainder(lda, K, A + x, newA + K * x, M % 8); //padding
    }

    int y = N - N % 8;
    for (int j = 0; j < y; j += 8) {
        transpose(lda, K, B + j * lda, newB + K * j); //transpose B to make it row major order.
    }

    if  (N % 8 > 0){
        transpose_remainder(lda, K, B + y * lda, newB + K * y, N%8); //padding
    }

    int NN = (N + 7) / 8 * 8;
    int MM = (M + 7) / 8 * 8;

    for (int j = 0; j < NN; j += 8) {
        // copy and transpose newB
        double* newB_ptr = newB + K * j;
        double* c_ptr = C + j * lda;
        /* For each row of A */
        for (int i = 0; i < MM; i += 8) {
            micro_kernal(lda, K, newA + i * K, newB_ptr, c_ptr);
            c_ptr += 8;
        }
    }
}

void multi_level_blocking(int lda, int ii, int jj, int kk, int MM, int NN, int KK, double *A, double *B, double *C) {
    for (int j = jj; j < NN; j += L1_SIZE) {
        // For each column j of B
        int N = min(L1_SIZE, NN - j);
        for (int i = ii; i < MM; i += L1_SIZE) {
            // For each row i of A
            int M = min(L1_SIZE, MM - i);
            for (int k = kk; k < KK; k += L1_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int K = min(L1_SIZE, KK - k);
                // Perform individual block dgemm
                do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *A, double *B, double *C) {
    init_maps(L1_SIZE);

    if (lda <= L1_SIZE){
        do_block(lda, lda, lda, lda, A, B, C);
    }else{
        for (int jj = 0; jj < lda; jj += L2_SIZE) {
            int NN = jj + min(L2_SIZE, lda - jj);
            for (int ii = 0; ii < lda; ii += L2_SIZE) {
                int MM = ii + min(L2_SIZE, lda - ii);
                for (int kk = 0; kk < lda; kk += L2_SIZE) {
                    int KK = kk + min(L2_SIZE, lda - kk);
                    multi_level_blocking(lda, ii, jj, kk, MM, NN, KK, A, B, C);
                }
            }
        }
    }

    free_maps();
}
