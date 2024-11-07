#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define FactorIntToDouble 1.1

// Global pointers for matrices to handle dynamic size
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
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            double resultValue = 0;
            for (int transNumber = 0; transNumber < N; transNumber++) {
                resultValue += firstMatrix[row][transNumber] * secondMatrix[transNumber][col];
            }
            matrixMultiResult[row][col] = resultValue;

            // Debug print to check which thread is executing
            #pragma omp critical
            {
                printf("Thread %d calculated matrixMultiResult[%d][%d] = %f\n", omp_get_thread_num(), row, col, resultValue);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    // Check for command-line argument for N
    if (argc != 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    // Set matrix size N from command line argument
    int N = atoi(argv[1]);
    if (N <= 0) {
        printf("Matrix size must be a positive integer.\n");
        return 1;
    }

    // Allocate memory for matrices
    allocateMatrices(N);

    // Initialize matrices
    printf("Initializing matrices of size %dx%d...\n", N, N);
    matrixInit(N);

    // Start the parallel matrix multiplication
    printf("Starting parallel matrix multiplication...\n");
    double t1 = omp_get_wtime();
    matrixMulti(N);
    double t2 = omp_get_wtime();

    // Print the execution time
    printf("Parallel time: %f seconds\n", t2 - t1);

    // Free allocated memory
    freeMatrices(N);

    return 0;
}
