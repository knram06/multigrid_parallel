#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "gauss_elim.h"

/* Solves 1D poisson with Dirichlet bcs as 0 at positions
 * 0 and 1 */


/* Solves simple poisson on 1D with rhs as cosx
 * Verify against analytical soln
 * Understand storage and strided access patterns
 * And areas for parallelism
 */

double func(double x)
{ return x;}
//{ return x*x/2;}
//{ return (-cos(x) + x*(cos(1)-1) + 1); }

double rhsFunc(double x)
{ return 0.;}
//{ return 1.;}
//{ return cos(x);}

int main(int argc, char** argv)
{
    if(argc != 4)
    {
        printf("Usage: %s <coarse grid points> <number of levels> <gauss seidel iterations>\n", argv[0]);
        exit(1);
    }

    // parse the passed in options
    int N = atoi(argv[1]);
    const int numLevels = atoi(argv[2]);
    const int gsIterNum = atoi(argv[3]);

    // preallocate the arrays using max grid level
    int multFactor = 1 << (numLevels-1);
    const int finestGridNum = ((N-1) * multFactor)+1;
    double *v = calloc(finestGridNum, sizeof(double)); // zero out array
    double *f = calloc(finestGridNum, sizeof(double)); // zero out array
    double *r = calloc(finestGridNum, sizeof(double)); // zero out array

    // enforce bcs and coarse grid value
    v[0] = func(0); v[finestGridNum-1] = func(1);

    // calculate fine spacing
    // update N
    N = finestGridNum;
    double h = 1./ (N-1);
    int i, j;

    // fill out f array - not necessary here
    // but just doing it for actual case
    for(i = 0; i < finestGridNum; i++)
        f[i] = rhsFunc(i*h);

    // Restriction part
    // from FINE to COARSE
    multFactor = 1; // set mf to corresponding to fine grid
    for(i = numLevels-1; i > 0; i--)
    {
        int p;
        const double hSq = (h*h);
        // PERF: do TILING here to improve performance?
        for(p = 0; p < gsIterNum; p++)
        {
            for(j = multFactor; j < (N-1)*multFactor; j += multFactor)
                v[j] = (v[j-multFactor] + v[j+multFactor] - hSq*f[j]) / 2;

        } // end of Gauss Seidel smoother

        // PERF: combine the two for loops below in some way?
        // Simplify the calculation?
        // evaluate residual
        for(j = multFactor; j < (N-1)*multFactor; j += multFactor)
            r[j] = f[j] - (v[j-multFactor] + v[j+multFactor] - 2*v[j])/hSq;

        // do restriction operator
        for(j = 2*multFactor; j < (N-1)*multFactor; j += 2*multFactor)
            f[j] = 0.25*(r[j-multFactor] + r[j+multFactor]) + 0.5*r[j];

        // update vars
        h *= 2; multFactor *= 2;
        N = (N+1)/2;
    }

    // if here, we are on the coarse grid - so do Direct Solve
    // Gaussian Elimination
    // prepare matrices to fill and send to G-S elimination
    {
        double *A = calloc(N*N, sizeof(double));
        double *b = calloc(N, sizeof(double));
        double *x = calloc(N, sizeof(double));

        // fill in the matrix - opposing sign convention
        // REMEMBER to adjust for sign of b then
        A[0] = 1;
        b[0] = 0;
        for(i = 1; i < N-1; i++)
        {
            const int nii = N*i + i;
            A[nii-1] = -1; A[nii] = 2; A[nii+1] = -1;
        }
        A[N*N-1] = 1;
        b[N-1] = 0;

        // obtain solution
        gaussianElimination(A, N, b, x);

        // map obtained solution back to finest grid
        for(i = 1; i < N-1; i++)
            v[i*multFactor] = x[i];

        free(x);
        free(b);
        free(A);
    }

    // Prolongation?
    // from COARSE to FINE
    for(i = 1; i < numLevels; i++)
    {
        h /= 2; N = 2*N-1;
        multFactor /= 2;

        int j;
        for(j = multFactor; j < (N-1)*multFactor; j += (2*multFactor))
            v[j] += (v[j-multFactor] + v[j+multFactor])/2;

        // simple cache to avoid multiple
        const double hSq = (h*h);
        // do Gauss seidel iteration
        int p;
        for(p = 0; p < gsIterNum; p++)
        {
            for(j = multFactor; j < (N-1)*multFactor; j += multFactor)
                v[j] = (v[j-multFactor] + v[j+multFactor] - hSq*rhsFunc(j*h)) / 2;

        } // end of for loop through p
    }

    // check against analytical soln of (-cos(x) + x*(cos1-1) + 1)
    double res = 0.0;

    for(i = 0; i < finestGridNum; i++)
    {
        double x = i*h;
        //double diff = v[i] - (-cos(x) + x*(cos(1)-1) + 1);
        //double diff = v[i] - (x*x)/2;
        double diff = v[i] - func(x);
        res += diff*diff;
    }

    printf("Error norm: %lf\n", res);

    free(r);
    free(f);
    free(v);
    return 0;
}

