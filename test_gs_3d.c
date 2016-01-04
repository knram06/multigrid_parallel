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

    // timing info
    double timingTemp;
    TimingInfo tInfo;
    tInfo.numCalls  = 0;
    tInfo.timeTaken = 0.;
    const double cycleTime = 1./CLOCKS_PER_SEC;

    double *u = calloc(N*N*N, sizeof(double));
    double *d = calloc(N*N*N, sizeof(double));

    // fill in the details at the finest level
    double h = GRID_LENGTH/(finestOneSideNum-1);

    // enforce the boundary conditions
    setupBoundaryConditions(u, finestOneSideNum, h);

    double norm = 1e9, tolerance = 1e-6;

    // compare against squared tolerance, can avoid
    // unnecessary sqrts that way?
    // RELATIVE CONVERGENCE criteria
    const double initResidual = calculateResidual(u, d, finestOneSideNum, h, NULL);

    /*FALSE SHARING issue below
     * let each thread write the residual in its portion to an array
     * To facilitate global NORM calculation
     * CAN'T use OMP REDUCTION here!!
     */

    const double cmpNorm = initResidual*tolerance;
    int iterCount = 1;
    double relResidualRatio = -1;
    double oldNorm = -1;

    while(norm > cmpNorm)
    {
        oldNorm = norm;
        timingTemp = omp_get_wtime();

        GaussSeidelSmoother(u, d, N, h, 1);

        tInfo.timeTaken += (omp_get_wtime() - timingTemp);
        tInfo.numCalls++;

        // calc residual reduction ratio
        norm = calculateResidual(u, d, finestOneSideNum, h, NULL);
        relResidualRatio = norm/oldNorm;
        printf("%5d    Residual Norm:%20g     ResidRatio:%20g\n", iterCount, norm, relResidualRatio);
        iterCount++;
    }
    //clock_t endTime = clock();
    //printf("Time taken: %lf\n", (endTime-timingTemp)*cycleTime);

    // print the timing info
    printf("Number of calls: %d\nTime taken:%lf\n", tInfo.numCalls, tInfo.timeTaken);

    writeOutputData("output.vtk", u, h, finestOneSideNum);

    // checking against analytical soln
    double errNorm = 0.;
    int i, j, k;
    for(i = 0; i < N; i++)
    {
        int nni = N*N*i;
        for(j = 0; j < N; j++)
        {
            int nj = N*j;
            for(k = 0; k < N; k++)
            {
                int pos = nni + nj + k;
                double diff = u[pos] - BCFunc(i*h, j*h, k*h);
                errNorm = diff*diff;
            }
        }
    }

    printf("Error norm: %lf\n", errNorm);

    free(d);
    free(u);
    //deAllocTimingInfo(&tInfo, numLevels);

    return 0;
}

