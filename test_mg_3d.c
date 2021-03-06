#include <stdio.h>
#include <string.h>

#define GRID_LENGTH (1.)
#include "mg_3d.h"
#include "postprocess.h"

int main(int argc, char** argv)
{

    SolverInitialize(argc, argv);

    double *grid = NULL, *rhs = NULL;
    double h;
    int finestOneSideNum = SolverGetDetails(&grid, &rhs, &h);

    SolverSetupBoundaryConditions();

    double norm = 1e9, tolerance = 1e-8;
    int numThreads = omp_get_max_threads();
    double *threadNorm = calloc(numThreads, sizeof(double));

    // compare against squared tolerance, can avoid
    // unnecessary sqrts that way?
    // RELATIVE CONVERGENCE criteria
    const double initResidual = SolverGetInitialResidual();

    // ENFORCE DIRICHLET on X vector
    setupBoundaryConditions(grid, finestOneSideNum, h);

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
                errNorm += diff*diff;
            }
        }
    }
    errNorm = sqrt(errNorm);
    printf("Error norm: %10.6g\n", errNorm);

    writeOutputData("diff2.vtk", grid, h, finestOneSideNum);

    SolverFinalize();
    free(threadNorm);

    return 0;
}

