#include <stdio.h>
#include <string.h>

#define GRID_LENGTH (1.)
#include "mg_3d.h"
#include "postprocess.h"

int main(int argc, char** argv)
{
    if(argc != 4)
    {
        printf("Usage: %s <coarse grid points on one side> <number of levels> <gauss seidel iterations>\n", argv[0]);
        exit(1);
    }

    // parse the passed in options
    int N = atoi(argv[1]);
    const int numLevels = atoi(argv[2]);
    const int gsIterNum = atoi(argv[3]);

    // preallocate the arrays using max grid level
    int multFactor = 1 << (numLevels-1);
    const int finestOneSideNum = ((N-1) * multFactor)+1;

    //TimingInfo **tInfo = NULL;
    // allocate the timing object
    allocTimingInfo(&tInfo, numLevels);

    double **u = NULL, **d = NULL, **r = NULL;
    allocGridLevels(&u, numLevels, N);
    allocGridLevels(&d, numLevels, N);
    allocGridLevels(&r, numLevels, N);

    // fill in the details at the finest level
    double h = GRID_LENGTH/(finestOneSideNum-1);

    // preallocate and fill the coarse matrix A
    int matDim = N*N*N;
    double *A = calloc(matDim*matDim, sizeof(double));
    constructCoarseMatrixA(A, N, h);
    convertToLU_InPlace(A, matDim);

    // enforce the boundary conditions
    setupBoundaryConditions(u[numLevels-1], finestOneSideNum, h);

    double norm = 1e9, tolerance = 1e-6;

    // compare against squared tolerance, can avoid
    // unnecessary sqrts that way?
    // RELATIVE CONVERGENCE criteria
    const double initResidual = calculateResidual(u[numLevels-1], d[numLevels-1], finestOneSideNum, h, NULL);

    const double cmpNorm = initResidual*tolerance;
    int iterCount = 1;
    double relResidualRatio = -1;
    double oldNorm = -1;
    while(norm >= cmpNorm)
    {
        oldNorm = norm;
        norm = vcycle(u, d, r, numLevels-1, numLevels, gsIterNum, finestOneSideNum, A);
        relResidualRatio = norm/oldNorm;
        //norm = calculateResidual(u[numLevels-1], d[numLevels-1], finestOneSideNum, h);
        printf("%5d    Residual Norm:%20g     ResidRatio:%20g\n", iterCount, norm, relResidualRatio);
        iterCount++;
    }

    // smoothen border edge and point values
    // although they are not used in the calculation
    //updateEdgeValues(u[numLevels-1], finestOneSideNum);

    printTimingInfo(tInfo, numLevels);

    writeOutputData("output.vtk", u[numLevels-1], h, finestOneSideNum);

    // checking against analytical soln
    double errNorm = 0.;
    int i, j, k;
    double *v = u[numLevels-1];
    for(i = 0; i < finestOneSideNum; i++)
    {
        int nni = finestOneSideNum*finestOneSideNum*i;
        for(j = 0; j < finestOneSideNum; j++)
        {
            int nj = finestOneSideNum*j;
            for(k = 0; k < finestOneSideNum; k++)
            {
                int pos = nni + nj + k;
                double diff = v[pos] - BCFunc(i*h, j*h, k*h);
                v[pos] = diff;
                errNorm = diff*diff;
            }
        }
    }
    errNorm = sqrt(errNorm);
    printf("Error norm: %lf\n", errNorm);

    deAllocGridLevels(&r, numLevels);
    deAllocGridLevels(&d, numLevels);
    deAllocGridLevels(&u, numLevels);

    deAllocTimingInfo(&tInfo, numLevels);
    free(A);

    return 0;
}

