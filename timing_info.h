#ifndef TIMING_INFO_H
#define TIMING_INFO_H

#define NUM_STAGES 7
#define NUM_ATTRIBS 2

#include <stdlib.h>
#include <time.h>
#include <assert.h>

typedef struct __time_t
{
    int numCalls;
    double timeTaken;
} TimingInfo;

void allocTimingInfo(TimingInfo ***tInfo, const int levels)
{
    (*tInfo) = malloc(sizeof(TimingInfo*) * levels);
    assert((*tInfo));

    int i;
    for(i = 0; i < levels; i++)
    {
        (*tInfo)[i] = malloc(sizeof(TimingInfo) * NUM_STAGES);
        assert((*tInfo)[i]);

        // initialize the object
        (*tInfo)[i]->numCalls = 0;
        (*tInfo)[i]->timeTaken = 0.;
    }
}

void resetTimingInfo(TimingInfo **tInfo, const int levels)
{
    int l, s;
    for(l = 0; l < levels; l++)
    {
        TimingInfo *t = tInfo[l];
        for(s = 0; s < NUM_STAGES; s++)
        {
            t[s].numCalls = 0;
            t[s].timeTaken = 0.;
        }
    } // end of levels loop
}

void printTimingInfo(TimingInfo **tInfo, const int levels)
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

void deAllocTimingInfo(TimingInfo ***tInfo, const int levels)
{
    int i;
    for(i = 0; i < levels; i++)
        free((*tInfo)[i]);

    free((*tInfo));
}

#endif
