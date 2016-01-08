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
    int maxThreads = omp_get_max_threads();
    printf("Max OMP threads: %d\n", omp_get_max_threads());
    double *threadNorm = calloc(maxThreads, sizeof(double));

    const double cmpNorm = initResidual*tolerance;
    int iterCount = 1;
    double relResidualRatio = -1;
    double oldNorm = -1;

    #pragma omp parallel
    {
        //printf("Number of threads: %d\n", omp_get_num_threads());
        int tid = omp_get_thread_num();
        //printf("tid: %d\n", tid);
    while(norm > cmpNorm)
    {
        #pragma omp single
        {
        oldNorm = norm;
        timingTemp = omp_get_wtime();
        }

        //GaussSeidelSmoother(u, d, N, h, 1);
        preSmoother(u, d, N, h, 1);
        postSmoother(u, d, N, h, 1);

        #pragma omp single
        {
        tInfo.timeTaken += (omp_get_wtime() - timingTemp);
        tInfo.numCalls++;
        }

        // FALSE SHARING - possible here
        //#pragma omp single
        threadNorm[tid] = calculateResidual(u, d, N, h, NULL);
        #pragma omp barrier     // VERY IMPORTANT!!

        #pragma omp single
        {
        int i;
        norm = 0.;
        for(i = 0; i < maxThreads; i++)
            norm += threadNorm[i]*threadNorm[i];

        // update with its sqrt
        norm = sqrt(norm);

        // calc residual reduction ratio
        relResidualRatio = norm/oldNorm;
        //norm = calculateResidual(u[numLevels-1], d[numLevels-1], finestOneSideNum, h);
        printf("%5d    Residual Norm:%20g     ResidRatio:%20g\n", iterCount, norm, relResidualRatio);
        iterCount++;
        }
    }
    } // end of PRAGMA OMP
    //clock_t endTime = clock();
    //printf("Time taken: %lf\n", (endTime-timingTemp)*cycleTime);

    // smoothen border edge and point values
    // although they are not used in the calculation
    //updateEdgeValues(u[numLevels-1], finestOneSideNum);

    // get some info
    printf("Max OMP threads: %d\n", omp_get_max_threads());
    // print the timing info
    printf("Number of calls: %d\nTime taken:%lf\n", tInfo.numCalls, tInfo.timeTaken);

    //writeOutputData("output.vtk", u, h, finestOneSideNum);

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

    printf("Error norm: %lf\n", sqrt(errNorm));

    free(d);
    free(u);
    free(threadNorm);
    //deAllocTimingInfo(&tInfo, numLevels);

    return 0;
}

