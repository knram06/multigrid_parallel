#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "gauss_elim.h"

/* Solves 1D poisson with Dirichlet bcs as 0 at positions
 * 0 and 1 */

void allocGridLevels(double ***u, const int numLevels, const int N)
{
    *u = malloc(sizeof(double*) * numLevels);

    // (*u) is IMPORTANT here
    // else seemed to overwrite top level stack parameters?
    assert(*u);
    int i;

    for(i = 0; i < numLevels; i++)
    {
        int numNodes = ((N-1)*(1 << i) + 1);
        (*u)[i] = calloc(numNodes, sizeof(double));
        assert((*u)[i]);
    }
}

void deAllocGridLevels(double ***u, const int numLevels)
{
    int i;
    for(i = 0; i < numLevels; i++)
        free((*u)[i]);

    free(*u);
}

double getSquaredNorm(double **u, double **f, int q, int N)
{
    double h = 1./(N-1);
    double hSqInv = 1./(h*h);

    double *v = u[q];
    double *d = f[q];

    int i;
    double res = 0.;

    // should this be from 0 to N instead?
    for(i = 1; i < N-1; i++)
    {
        double diff = d[i] - (v[i-1] + v[i+1] - 2*v[i])*hSqInv;
        res += diff*diff;
    }

    return res;
}

void GaussSeidelSmoother(double *v, double *d, const int N, const double h, const int smootherIter)
{
    int i, j;
    const double hSq = h*h;
    // do pre-smoother first
    // PERF: tile here?
    for(i = 0; i < smootherIter; i++)
    {
        for(j = 1; j < N-1; j++)
            v[j] = (v[j-1] + v[j+1] - hSq*d[j]) / 2.;
    }
}

void multigrid_method(double **u, double **f, int q, const int smootherIter, int N)
{
    int i, j;
    if(q == 0)
    {
        // prepare A matrix to send to gaussian elimination
        double *A = calloc(N*N, sizeof(double));
        A[0] = 1.; A[N*N-1] = 1.;

        for(i = 1; i < N-1; i++)
        {
            const int ni = (N+1)*i;
            A[ni-1] = 1; A[ni] = -2; A[ni+1] = 1;
        }

        gaussianElimination(A, N, f[q], u[q]);

        free(A);
        return;
    }

    double h = 1./(N-1);
    double hSq = h*h;

    double *v = u[q];
    double *d = f[q];

    GaussSeidelSmoother(v, d, N, h, smootherIter);

    // allocate the residual vector
    double *r = malloc(sizeof(double) * N);
    r[0] = 0; r[N-1] = 0;

    // evaluate the residual
    for(j = 1; j < N-1; j++)
        r[j] = hSq*d[j] - (v[j-1] + v[j+1] - 2*v[j]);

    // update N for next coarser level
    int N_coarse = (N+1)/2;

    // now restrict this onto the next level
    double *d1 = f[q-1];
    for(j = 1; j < N_coarse-1; j++)
        d1[j] = 0.5*r[2*j] + 0.25*(r[2*j-1] + r[2*j+1]);

    // do recursive call now
    multigrid_method(u, f, q-1, smootherIter, N_coarse);

    // now do prolongation to the fine grid
    double *v1 = u[q-1];
    // reuse residual array
    // copy into r_2j
    // PARALLELIZABLE
    r[N-1] = v1[N_coarse-1];
    for(j = 0; j < N_coarse-1; j++)
    {
        r[2*j] = v1[j];
        r[2*j+1] = 0.5 * (v1[j] + v1[j+1]);
    }

    // above is the effective error
    // so do coarse grid correction now
    // PARALLELIZABLE
    for(j = 0; j < N; j++)
        v[j] += r[j];

    // do more smoother iterations here - put it into a function?
    GaussSeidelSmoother(v, d, N, h, smootherIter);

    // free the residual memory used
    free(r);
}


/* Solves simple poisson on 1D with rhs as cosx
 * Verify against analytical soln
 * Understand storage and strided access patterns
 * And areas for parallelism
 */

double func(double x)
{ return (-cos(x) + x*(cos(1)-1) + 1); }
//{ return 0.5*(x*x - x); }
//{ return x;}
//{ return x*x/2;}

double rhsFunc(double x)
{ return cos(x);}
//{ return 1.;}
//{ return 0.;}
//{ return 1.;}

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

    double **u = NULL, **d = NULL;
    allocGridLevels(&u, numLevels, N);
    allocGridLevels(&d, numLevels, N);

    int i;
    // fill in the details at the finest level
    double h = 1./(finestGridNum-1);
    for(i = 0; i < finestGridNum; i++)
        d[numLevels-1][i] = rhsFunc(i*h);

    // enforce bcs at finest level
    u[numLevels-1][0]               = func(0);
    u[numLevels-1][finestGridNum-1] = func(1);

    double norm = 1e9, tolerance = 1e-6;

    // compare against squared tolerance, can avoid
    // unnecessary sqrts that way?
    while(norm >= tolerance*tolerance)
    {
        multigrid_method(u, d, numLevels-1, gsIterNum, finestGridNum);
        norm = getSquaredNorm(u, d, numLevels-1, finestGridNum);
        printf("norm: %g\n", norm);
    }

    // checking against analytical soln
    double errNorm = 0.;
    for(i = 0; i < finestGridNum; i++)
    {
        double diff = u[numLevels-1][i] - func(i*h);
        errNorm = diff*diff;
    }

    printf("Error norm: %lf\n", errNorm);

    deAllocGridLevels(&d, numLevels);
    deAllocGridLevels(&u, numLevels);

    return 0;
}

