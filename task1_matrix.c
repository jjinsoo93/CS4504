#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define FactorIntToDouble 1.1

double **firstMatrix;
double **secondMatrix;
double **matrixMultiResult;

void allocateMatrices(int N) {
    firstMatrix = (double **)malloc(N * sizeof(double *));
    secondMatrix = (double **)malloc(N * sizeof(double *));
    matrixMultiResult = (double **)malloc(N * sizeof(double *));
    for (int i = 0; i < N; i++) {
        firstMatrix[i] = (double *)malloc(N * sizeof(double));
        secondMatrix[i] = (double *)malloc(N * sizeof(double));
        matrixMultiResult[i] = (double *)malloc(N * sizeof(double));
    }
}

void freeMatrices(int N) {
    for (int i = 0; i < N; i++) {
        free(firstMatrix[i]);
        free(secondMatrix[i]);
        free(matrixMultiResult[i]);
    }
    free(firstMatrix);
    free(secondMatrix);
    free(matrixMultiResult);
}

void matrixInit(int N) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            srand(row + col); // Initialize random seed with a combination of row and col
            firstMatrix[row][col] = (rand() % 10) * FactorIntToDouble;
            secondMatrix[row][col] = (rand() % 10) * FactorIntToDouble;
        }
    }
}

void matrixMulti(int N) {
    int desired_threads = 8; // Adjust this based on your system capability
    omp_set_num_threads(desired_threads);
    printf("Using %d threads for matrix multiplication\n", desired_threads);

    #pragma omp parallel for schedule(dynamic)
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            double resultValue = 0;
            for (int transNumber = 0; transNumber < N; transNumber++) {
                resultValue += firstMatrix[row][transNumber] * secondMatrix[transNumber][col];
            }
            matrixMultiResult[row][col] = resultValue;
        }
        
        // Debug print to check which thread is processing each row
        #pragma omp critical
        {
            printf("Thread %d completed row %d\n", omp_get_thread_num(), row);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    if (N <= 0) {
        printf("Matrix size must be a positive integer.\n");
        return 1;
    }

    // Print max threads available for debug
    printf("Max threads available: %d\n", omp_get_max_threads());

    allocateMatrices(N);
    printf("Initializing matrices of size %dx%d...\n", N, N);
    matrixInit(N);

    printf("Starting parallel matrix multiplication...\n");
    double t1 = omp_get_wtime();
    matrixMulti(N);
    double t2 = omp_get_wtime();

    printf("Parallel time: %f seconds\n", t2 - t1);
    freeMatrices(N);

    return 0;
}
