#include <stdio.h>
#include <string.h>

#define GRID_LENGTH (1.)
#include "mg_3d.h"
#include "postprocess.h"

int main(int argc, char** argv)
{

    SolverInitialize(argc, argv);

    /*
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
    */

    double *grid = NULL, *rhs = NULL;
    double h;
    SolverGetDetails(&grid, &rhs, &h);

    SolverSetupBoundaryConditions();

    double norm = 1e9, tolerance = 1e-8;
    int numThreads = omp_get_max_threads();
    double *threadNorm = calloc(numThreads, sizeof(double));

    // compare against squared tolerance, can avoid
    // unnecessary sqrts that way?
    // RELATIVE CONVERGENCE criteria
    const double initResidual = SolverGetInitialResidual();

    const double cmpNorm = initResidual*tolerance;
    int iterCount = 1;
    double relResidualRatio = -1;
    double oldNorm = -1;

    double clockStart = omp_get_wtime();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        while(norm > cmpNorm)
        {
            oldNorm = norm;

            // POTENTIAL FALSE SHARING issue here
            threadNorm[tid] = SolverLinSolve();

            #pragma omp barrier // VERY IMPORTANT!!
            // let one thread calculate the actual norm
            #pragma omp single
            {
                int i;
                norm = 0;
                for(i = 0; i < numThreads; i++)
                {
                    // square and sum it to get the l2-norm
                    // at the end
                    norm += threadNorm[i]*threadNorm[i];
                }
                norm = sqrt(norm);

                relResidualRatio = norm/oldNorm;
                //norm = calculateResidual(u[numLevels-1], d[numLevels-1], finestOneSideNum, h);
                printf("%5d    Residual Norm:%20g     ResidRatio:%20g\n", iterCount, norm, relResidualRatio);
                iterCount++;
            }
        }
    } // end of PRAGMA OMP
    double clockEnd = omp_get_wtime();

    // smoothen border edge and point values
    // although they are not used in the calculation
    //updateEdgeValues(u[numLevels-1], finestOneSideNum);

    SolverPrintTimingInfo();
    printf("Max threads: %d\n", numThreads);
    printf("Overall time for solving: %10.6g\n", clockEnd-clockStart);

    // checking against analytical soln
    double errNorm = 0.;
    int i, j, k;
    for(i = 0; i < finestOneSideNum; i++)
    {
        int nni = finestOneSideNum*finestOneSideNum*i;
        for(j = 0; j < finestOneSideNum; j++)
        {
            int nj = finestOneSideNum*j;
            for(k = 0; k < finestOneSideNum; k++)
            {
                int pos = nni + nj + k;
                double diff = grid[pos] - BCFunc(i*h, j*h, k*h);
                grid[pos] = diff;
                errNorm = diff*diff;
            }
        }
    }
    errNorm = sqrt(errNorm);
    printf("Error norm: %10.6g\n", errNorm);

    //writeOutputData("diff2.vtk", v, h, finestOneSideNum);

    SolverFinalize();
    free(threadNorm);

    return 0;
}

