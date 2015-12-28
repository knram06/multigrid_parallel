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
    clock_t timingTemp;
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

    const double cmpNorm = initResidual*tolerance;
    int iterCount = 1;
    double relResidualRatio = -1;
    double oldNorm = -1;

    timingTemp = clock();
    while(norm >= cmpNorm)
    {
        oldNorm = norm;

        GaussSeidelSmoother(u, d, N, h, 1);
        //tInfo.timeTaken += (clock() - timingTemp);
        //tInfo.numCalls++;

        norm = calculateResidual(u, d, N, h, NULL);
        relResidualRatio = norm/oldNorm;
        //norm = calculateResidual(u[numLevels-1], d[numLevels-1], finestOneSideNum, h);
        printf("%5d    Residual Norm:%20g     ResidRatio:%20g\n", iterCount, norm, relResidualRatio);
        iterCount++;
    }
    clock_t endTime = clock();
    printf("Time taken: %lf\n", (endTime-timingTemp)*cycleTime);

    // smoothen border edge and point values
    // although they are not used in the calculation
    //updateEdgeValues(u[numLevels-1], finestOneSideNum);

    // get some info
    printf("Max OMP threads: %d\n", omp_get_max_threads());
    // print the timing info
    //printf("Number of calls: %d\nTime taken:%lf\n", tInfo.numCalls, tInfo.timeTaken*cycleTime);

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
    //deAllocTimingInfo(&tInfo, numLevels);

    return 0;
}

