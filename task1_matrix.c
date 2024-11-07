#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>

#define N 2048
#define FactorIntToDouble 1.1

double firstMatrix[N][N] = {0.0};
double secondMatrix[N][N] = {0.0};
double matrixMultiResult[N][N] = {0.0};

void matrixMulti()
{
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            double resultValue = 0;
            for (int transNumber = 0; transNumber < N; transNumber++) {
                resultValue += firstMatrix[row][transNumber] * secondMatrix[transNumber][col];
            }
            matrixMultiResult[row][col] = resultValue;
        }
    }
}

void matrixInit()
{
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            srand(row + col);
            firstMatrix[row][col] = (rand() % 10) * FactorIntToDouble;
            secondMatrix[row][col] = (rand() % 10) * FactorIntToDouble;
        }
    }
}

int main()
{
    printf(N);
    matrixInit();

    double t1 = omp_get_wtime();
    matrixMulti();
    double t2 = omp_get_wtime();
    
    printf("Parallel time: %f seconds\n", t2 - t1);

    return 0;
}