#ifndef TIMING_INFO_H
#define TIMING_INFO_H

#include <string.h>

typedef struct __time_t
{
    int numStages;
    char **stageNames;
    int  *numCalls;
    double *timeTaken;
} TimingInfo;

void allocTimingInfo(TimingInfo **tInfo, char **stageNames, const int numStages)
{
    (*tInfo) = malloc(sizeof(TimingInfo));
    assert((*tInfo));

    // set the number of stages
    (*tInfo)->numStages  = numStages;

    // copy over the stage names
    // IMPORTANT: free this later!
    (*tInfo)->stageNames = malloc(sizeof(char*) * numStages);
    int i;
    for(i = 0; i < numStages; i++)
        (*tInfo)->stageNames[i] = strdup(stageNames[i]);

    // set other attribs - FREE them in dealloc
    (*tInfo)->numCalls  = calloc(numStages, sizeof(int));
    (*tInfo)->timeTaken = calloc(numStages, sizeof(double));
}

void resetTimingInfo(TimingInfo *tInfo)
{
    memset(tInfo->numCalls,  0, sizeof(int)*tInfo->numStages);
    memset(tInfo->timeTaken, 0, sizeof(double)*tInfo->numStages);
}

void printTimingInfo(TimingInfo *tInfo)
{
    int i;
    printf("%20s %20s %20s\n", "", "numCalls", "timeTaken");

    for(i = 0; i < tInfo->numStages; i++)
        printf("%20.20s %20d %20lf\n", tInfo->stageNames[i], tInfo->numCalls[i], tInfo->timeTaken[i]);
}

/*
void printTimingInfo(TimingInfo *tInfo)
{
    int l, s;

    const char* stageNames[NUM_STAGES] = {
        "Smoother1", "CalcResidual1", "Restrict Residual", "Recurse, Direct Solve", "Prolongate&Correct", "Smoother2", "CalcResidual2"};

    for(l = 0; l < levels; l++)
    {
        const TimingInfo* t = tInfo[l];
        printf("LEVEL %d\n", l);
        printf("%20s %20s %20s\n", "", "numCalls", "timeTaken");

        for(s = 0; s < NUM_STAGES; s++)
            printf("%20.20s %20d %20lf\n", stageNames[s], t[s].numCalls, t[s].timeTaken);
    }
}
*/

void deAllocTimingInfo(TimingInfo **tInfo)
{
    int i;
    for(i = 0; i < (*tInfo)->numStages; i++)
        free( (*tInfo)->stageNames[i] );
    free((*tInfo)->stageNames);

    free((*tInfo)->numCalls);
    free((*tInfo)->timeTaken);

    free((*tInfo));
}

#endif
