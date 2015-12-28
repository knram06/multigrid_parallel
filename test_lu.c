#include <stdio.h>
#include <string.h>

#define GRID_LENGTH (1.)
#include "mg_3d.h"
#include "postprocess.h"

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        printf("Usage: %s <number of points>\n", argv[0]);
        exit(1);
    }

    // parse the passed in options
    finestOneSideNum  = atoi(argv[1]);
    int N = finestOneSideNum;

    // fill in the details at the finest level
    double h = GRID_LENGTH/(finestOneSideNum-1);

    // construct and allocate for the matrix
    int totalNodes = N*N*N;
    int matDim = totalNodes*totalNodes;
    double *A = calloc(matDim, sizeof(double));
    constructCoarseMatrixA(A, N, h);
    convertToLU_InPlace(A, totalNodes);

    double *u = calloc(totalNodes, sizeof(double));
    double *d = calloc(totalNodes, sizeof(double));

    // enforce the boundary conditions
    setupBoundaryConditions(d, finestOneSideNum, h);

    // do solve with LU
    clock_t start = clock();
    solveWithLU(A, totalNodes, d, u);
    clock_t diff = (clock() - start);

    printf("Time taken for LU solve: %lf\n", diff/(double)CLOCKS_PER_SEC);

    writeOutputData("output.vtk", u, h, finestOneSideNum);
    // checking against analytical soln
    //double errNorm = 0.;
    //for(i = 0; i < finestGridNum; i++)
    //{
    //    double diff = u[numLevels-1][i] - func(i*h);
    //    errNorm = diff*diff;
    //}

    //printf("Error norm: %lf\n", errNorm);

    free(d);
    free(u);
    free(A);
    //deAllocTimingInfo(&tInfo, numLevels);

    return 0;
}

