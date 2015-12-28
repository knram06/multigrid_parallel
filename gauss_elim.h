#ifndef GAUSS_ELIM_H
#define GAUSS_ELIM_H

#include <stdio.h>
//#define NUM (3)

// converts a given square matrix of size n
// to its LU form, reusing storage
void convertToLU_InPlace(double *a, int n)
{
    int i, j, k;

    // loop across rows
    for(i = 0; i < n-1; i++)
    {
        const int ni = n*i;
        const double aii_inv = 1./a[ni+i];

        for(k = i+1; k < n; k++)
        {
            const int nk = n*k;
            const double z = a[nk + i]*aii_inv;
            a[nk + i] = z;

            for(j = i+1; j < n; j++)
                a[nk + j] -= z*a[ni + j];
        }
    } // end of i loop
}

void solveWithLU(const double* __restrict__ LU, const int n, const double* __restrict__ b, double* __restrict__ x)
{
    int i, j;

    // solve Lz=b system - FORWARD Substitution
    // reuse x vector as z
    for(i = 0; i < n; i++)
    {
        const int ni = n*i;
        double sum = 0.;
        for(j = 0; j < i; j++)
            sum += LU[ni + j]*x[j];

        // implicitly L is unit lower triangular
        // so no need to divide
        x[i] = b[i] - sum;
    }

    // now solve Ux = z system - backward substitution
    // reuse x
    for(i = n-1; i >= 0; i--)
    {
        const int ni = n*i;
        double sum = 0.;
        for(j = n-1; j > i; j--)
            sum += LU[ni + j] * x[j];

        x[i] = (x[i]-sum)/LU[ni+i];
    }
}

// simple Gaussian elimination function
// not handling zero pivots and the like
// Laplacian equation when filled in the right order will not give 0 diagonals
void gaussianElimination(double *a, int n, double *b, double *res)
{
    int i, k;

    // loop across rows
    for(k = 0; k < n-1; k++)
    {
        for(i = k+1; i < n; i++)
        {
            const double z = a[n*i + k] / a[n*k + k];
            a[n*i + k] = 0.;

            // update b
            b[i] -= z*b[k];

            int j;
            for(j = k+1; j < n; j++)
                a[n*i+j] -= z*a[n*k + j];
        } // end of i loop across columns
    } // end of k loop across rows

    // now do back substituition
    for( i = n-1; i >= 0; i--)
    {
        double sum = 0.;
        int j;

        for(j = i+1; j < n; j++)
            sum += a[n*i+j]*res[j];

        res[i] = (b[i]-sum) / a[n*i + i];
    }
}

/*
int main()
{
    double b[NUM] = {3, 4, 5};
    double a[NUM*NUM] = {
         2, -1,  0,
        -1,  2, -1,
         0, -1,  2,
    };

    double res[NUM] = {0, 0, 0};

    convertToLU_InPlace(a, NUM);
    solveWithLU(a, NUM, b, res);
    //gaussianElimination(a, NUM, b, res);

    // print res vector
    printf("res:\n");
    int i;
    for(i = 0; i < NUM; i++)
        printf("%lf ", res[i]);
    printf("\n");

    return 0;
}
*/
#endif
