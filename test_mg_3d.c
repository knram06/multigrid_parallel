#include <stdio.h>
#include <string.h>

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

    double **u = NULL, **d = NULL;
    allocGridLevels(&u, numLevels, N);
    allocGridLevels(&d, numLevels, N);

    // preallocate and fill the coarse matrix A
    int matDim = N*N*N;
    double *A = calloc(matDim*matDim, sizeof(double));
    constructCoarseMatrixA(A, N);
    convertToLU_InPlace(A, matDim);

    // fill in the details at the finest level
    double h = GRID_LENGTH/(finestOneSideNum-1);

    // enforce the boundary conditions
    setupBoundaryConditions(u, finestOneSideNum, h, numLevels-1);

    double norm = 1e9, tolerance = 1e-6;

    // compare against squared tolerance, can avoid
    // unnecessary sqrts that way?
    // RELATIVE CONVERGENCE criteria
    const double initResidual = calculateResidual(u[numLevels-1], d[numLevels-1], finestOneSideNum, h, NULL);

    const double cmpNorm = initResidual*tolerance;
    int iterCount = 1;
    while(norm >= cmpNorm)
    {
        norm = vcycle(u, d, numLevels-1, numLevels-1, gsIterNum, finestOneSideNum, A);
        //norm = calculateResidual(u[numLevels-1], d[numLevels-1], finestOneSideNum, h);
        printf("%5d    Residual Norm:%20g\n", iterCount, norm);
        iterCount++;
    }

    // smoothen border edge and point values
    // although they are not used in the calculation
    //updateEdgeValues(u[numLevels-1], finestOneSideNum);

    printTimingInfo(tInfo, numLevels);

    writeOutputData("output.vtk", u[numLevels-1], h, finestOneSideNum);

    // checking against analytical soln
    //double errNorm = 0.;
    //for(i = 0; i < finestGridNum; i++)
    //{
    //    double diff = u[numLevels-1][i] - func(i*h);
    //    errNorm = diff*diff;
    //}

    //printf("Error norm: %lf\n", errNorm);

    deAllocGridLevels(&d, numLevels);
    deAllocGridLevels(&u, numLevels);

    deAllocTimingInfo(&tInfo, numLevels);
    free(A);

    return 0;
}

