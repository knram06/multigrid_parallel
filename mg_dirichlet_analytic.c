#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>

#include "gauss_elim.h"
#include "postprocess.h"
#include "timing_info.h"

#define GRID_LENGTH (1.)

TimingInfo **tInfo = NULL;
int coarseGridNum;
int finestOneSideNum;
int numLevels;
int gsIterNum;
bool useFMG;

// MG-level data
double **u, **d;
double *A; // coarsest level matrix
double spacing;  // spacing

double BCFunc(double x, double y, double z)
{return x*x -2*y*y + z*z;}
//{return 1;}
//{ return (-cos(x) + x*(cos(1)-1) + 1); }
//{ return 0.5*(x*x - x); }
//{ return x;}
//{ return x*x/2;}

//double rhsFunc(double x, double y, double z)
//{ return 0.;}
//{ return cos(x);}
//{ return 1.;}
//{ return 1.;}
void constructCoarseMatrixA(double *A, int N);

void allocGridLevels(double ***u, const int numLevels, const int N)
{
    *u = malloc(sizeof(double*) * numLevels);

    // (*u) is IMPORTANT here
    // else seemed to overwrite top level stack parameters?
    assert(*u);
    int i;

    for(i = 0; i < numLevels; i++)
    {
        int numNodes = ((N-1)*(1 << i) + 1);

        // let's say we do i,j,k order - so contiguous in k
        (*u)[i] = calloc(numNodes*numNodes*numNodes, sizeof(double));
        assert((*u)[i]);

    } // end of loop which sets up levels
}

// taken from
// http://stackoverflow.com/questions/600293/how-to-check-if-a-number-is-a-power-of-2
bool isPowerOfTwo(int x)
{ return (x & (x-1)) == 0;}

void SolverInitialize(int argc, char **argv)
{
    if(argc != 5)
    {
        printf("Usage: %s <coarse grid points on one side> <number of levels> <gauss seidel iterations> <useFMG?>\n", argv[0]);
        exit(1);
    }

    // parse the passed in options
    coarseGridNum = atoi(argv[1]);
    numLevels = atoi(argv[2]);
    gsIterNum = atoi(argv[3]);
    useFMG = (bool)(atoi(argv[4]));

    // assert that (coarseGridNum-1) is a power of 2
    // this ensures that coarsest grid can capture the capillary
    // But this breaks generality?!
    assert( isPowerOfTwo(coarseGridNum-1) );

    // preallocate the arrays using max grid level
    int multFactor = 1 << (numLevels-1);
    finestOneSideNum = ((coarseGridNum-1) * multFactor)+1;

    u = NULL; d = NULL;
    allocGridLevels(&u, numLevels, coarseGridNum);
    allocGridLevels(&d, numLevels, coarseGridNum);

    // allocate the timing object
    allocTimingInfo(&tInfo, numLevels);

    // fill in the details at the finest level
    spacing = GRID_LENGTH/(finestOneSideNum-1);

    // preallocate and fill the coarse matrix A
    int matDim = coarseGridNum*coarseGridNum*coarseGridNum;
    A = calloc(matDim*matDim, sizeof(double));
    constructCoarseMatrixA(A, coarseGridNum);
    convertToLU_InPlace(A, matDim);

}

// assume A has been preallocated
void constructCoarseMatrixA(double *A, int N)
{
    int i, j, k;
    const int NN = N*N;
    const int totalNodes = NN*N;

    // CORRECT for the construction of A matrix by
    // dividing by h^2 (Thanks to Rajesh Gandham for pointing this out)
    double h = GRID_LENGTH/(N-1);
    double hSq = h*h;
    double invHsq = 1./hSq;

    const double oneCoeff = 1.*invHsq;
    const double sixCoeff = 6.*invHsq;

    // just in case
    assert(totalNodes*totalNodes < INT_MAX);

    for(i = 0; i < N; i++)
    {
        const int nni = NN*i;
        for(j = 0; j < N; j++)
        {
            const int nj = N*j;
            for(k = 0; k < N; k++)
            {
                int pos = nni + nj + k;
                int mat1DIndex = (totalNodes+1)*pos;

                if(k == 0 || k == N-1
                || j == 0 || j == N-1
                || i == 0 || i == N-1)
                    A[mat1DIndex]    = 1;
                else
                {
                    // i-1,                      i+1
                    A[mat1DIndex-NN] = oneCoeff; A[mat1DIndex+NN] = oneCoeff;
                    // j-1,                      j+1
                    A[mat1DIndex-N]  = oneCoeff; A[mat1DIndex+N]  = oneCoeff;
                    // k-1,                      k+1
                    A[mat1DIndex-1]  = oneCoeff; A[mat1DIndex+1]  = oneCoeff;

                    // node (i,j,k)
                    A[mat1DIndex]    = -sixCoeff;
                }
            } // end of k loop
        } // end of j loop
    } // end of i loop
}

int SolverGetDetails(double **grid, double *h)
{
    // set the user pointer to finest level
    (*grid) = u[numLevels-1];

    *h = spacing;
    return finestOneSideNum;
}

void deAllocGridLevels(double ***u, const int numLevels)
{
    int i;
    for(i = 0; i < numLevels; i++)
        free((*u)[i]);

    free(*u);
}

// smoother function
void GaussSeidelSmoother(double* __restrict__ v, const double* __restrict__ d, const int N, const double h, const int smootherIter)
{
    int s;
    int i, j, k;
    const double hSq = h*h;

    const double invMultFact = 1./6;

    double center[2] = {GRID_LENGTH/2., GRID_LENGTH/2.};
    const int NN = N*N;

    // do pre-smoother first
    // PERF: tile here?
    for(s = 0; s < smootherIter; s++)
    {
        for(i = 1; i < N-1; i++)
        {
            const int nni = NN*i;
            for(j = 1; j < N-1; j++)
            {
                const int nj = N*j;
                int pos = nni + nj;
                for(k = 1; k < N-1; k++)
                {
                    int p = pos+k; // effectively nni+nj+k
                    v[p] = invMultFact*(
                            v[p - NN] + v[p + NN] // u[i-1] + u[i+1]
                          + v[p - N]  + v[p + N]  // u[j-1] + u[j+1]
                          + v[p - 1]  + v[p + 1]  // u[k-1] + u[k+1]
                          - hSq*d[p]              // hSq*f
                            );

                    /*
                    // enforce Neumann bc (order?)
                    // if on the inner node adjacent to boundary
                    // copy to boundary node - this way we ensure residual
                    // is zero on boundary node
                    if(i == 1 || i == N-2)
                    {
                        double ty = j*h - center[0];
                        double tz = k*h - center[1];
                        double rr = ty*ty + tz*tz;

                        if(i == 1)
                        {
                            // outside capillary radius
                            if (rr > CAPILLARY_RADIUS*CAPILLARY_RADIUS)
                            {
                                // copy (i,j,k) to (i-1,j,k)
                                v[p-NN] = v[p];
                            }
                        } // end of if i==1
                        // if i==N-2
                        else
                        {
                            // outside annular ring
                            if(rr <= (EXTRACTOR_INNER_RADIUS*EXTRACTOR_INNER_RADIUS)
                                    ||
                               rr >= (EXTRACTOR_OUTER_RADIUS*EXTRACTOR_OUTER_RADIUS))
                            {
                                // copy (i,j,k) to (i+1,j,k)
                                v[p+NN] = v[p];
                            }
                        } // end of else, i.e. i == N-2
                    } // end of if on X faces

                    // if on Y-Faces
                    if(j == 1)
                    {
                        // copy (i,j,k) to (i,j-1,k)
                        v[p-N] = v[p];
                    }
                    else if(j == N-2)
                    {
                        // (i,j,k) to (i,j+1,k)
                        v[p+N] = v[p];
                    }

                    // if on Z-Faces
                    if(k == 1)
                        v[p-1] = v[p];
                    else if(k == N-2)
                        v[p+1] = v[p];
                    */
                } // end of k loop
            } // end of j loop
        } // end of i loop
    } // end of smootherIter loop

} // end of GaussSeidelSmoother

double calculateResidual(const double* __restrict__ v, const double* __restrict__ d, const int N, const double h, double *res)
{
    int i, j, k;
    const double invHsq = 1./(h*h);
    const int NN = N*N;

    // adjust for different boundary condition types?
    double ret = 0.;
    for(i = 1; i < N-1; i++)
    {
        const int nni = NN*i;
        for(j = 1; j < N-1; j++)
        {
            const int nj = N*j;
            int pos = nni + nj;
            for(k = 1; k < N-1; k++)
            {
                // effectively (nni+nj+k)
                int p = pos + k;
                double diff = d[p] - invHsq*(v[p-NN] + v[p+NN] +
                                             v[p-N]  + v[p+N]  +
                                             v[p-1]  + v[p+1]  - 6*v[p]);

                // fill in array if it exists
                if(res)
                    res[p] = diff;

                ret += diff*diff;
            }
        } // end of j loop
    } // end of i loop
    return sqrt(ret);
}

void restrictResidual(const double* __restrict__ r, const int Nf, double* __restrict__ d, const int Nc)
{
    int i, j, k;
    const int NCNC = Nc*Nc;
    const int NFNF = Nf*Nf;

    // explicitly store nodal weights
    const double nodalWeights[3][3][3] = {
        // (x,y)=(0,0) line
        {
            // k varies from 0 to 2
            {0.015625, 0.03125, 0.015625},  // (1/64, 1/32, 1/64)
            { 0.03125,  0.0625,  0.03125},  // (1/32, 1/16, 1/32)
            {0.015625, 0.03125, 0.015625}   // (1/64, 1/32, 1/64)
        },
        // (x,y)=(1,0) line
        {
            {0.03125, 0.0625, 0.03125},     // (1/32, 1/16, 1/32)
            { 0.0625,  0.125,  0.0625},     // (1/16, 1/8,  1/16)
            {0.03125, 0.0625, 0.03125},     // (1/32, 1/16, 1/32))
        },
        // (x,y)=(2,0) line
        {
            // k varies from 0 to 2
            {0.015625, 0.03125, 0.015625},  // (1/64, 1/32, 1/64)
            { 0.03125,  0.0625,  0.03125},  // (1/32, 1/16, 1/32)
            {0.015625, 0.03125, 0.015625}   // (1/64, 1/32, 1/64)
        }
    };

    // for boundary faces
    // simply copy over coarse to fine nodal points
    int nnic, njc;
    int nnif, njf;

    /**********************************************/
    // X faces
    i = 0;
    for(j = 0; j < Nc; j++)
    {
        njc = j*Nc;
        njf = 2*j*Nf;
        for(k = 0; k < Nc; k++)
            d[njc + k] = r[njf + 2*k];
    }

    i = Nc-1;
    nnic = NCNC*i;
    nnif = NFNF*2*i;
    for(j = 0; j < Nc; j++)
    {
        njc = j*Nc;
        njf = 2*j*Nf;
        for(k = 0; k < Nc; k++)
            d[nnic + njc + k] = r[nnif + njf + 2*k];
    }
    /**********************************************/
    /**********************************************/
    // Y faces
    j = 0;
    njc = j*Nc;
    njf = 2*j*Nf;
    for(i = 0; i < Nc; i++)
    {
        nnic = NCNC*i;
        nnif = NFNF*2*i;
        for(k = 0; k < Nc; k++)
            d[nnic + njc + k] = r[nnif + njf + 2*k];
    }

    j = Nc-1;
    njc = j*Nc;
    njf = 2*j*Nf;
    for(i = 0; i < Nc; i++)
    {
        nnic = NCNC*i;
        nnif = NFNF*2*i;
        for(k = 0; k < Nc; k++)
            d[nnic + njc + k] = r[nnif + njf + 2*k];
    }
    /**********************************************/
    /**********************************************/
    // Z faces
    k = 0;
    for(i = 0; i < Nc; i++)
    {
        nnic = NCNC*i;
        nnif = NFNF*2*i;
        for(j = 0; j < Nc; j++)
        {
            njc = Nc*j;
            njf = Nf*2*j;
            d[nnic + njc + k] = r[nnif + njf + 2*k];
        }
    }

    k = Nc-1;
    for(i = 0; i < Nc; i++)
    {
        nnic = NCNC*i;
        nnif = NFNF*2*i;
        for(j = 0; j < Nc; j++)
        {
            njc = Nc*j;
            njf = Nf*2*j;
            d[nnic + njc + k] = r[nnif + njf + 2*k];
        }
    }
    /**********************************************/

    // now do interpolation for inner nodes
    for(i = 1; i < Nc-1; i++)
    {
        nnic = NCNC*i;
        nnif = NFNF*2*i;
        for(j = 1; j < Nc-1; j++)
        {
            njc = Nc*j;
            njf = Nf*2*j;

            for(k = 1; k < Nc-1; k++)
            {
                double val = 0.;

                // now we are at point (2*ic,2*jc,2*kc)
                // lower corner of cube will be (if-1,jf-1,kf-1) (on fine grid)
                int newPos = (nnif-NFNF) + (njf-Nf) + (2*k-1);

                int ti, tj, tk;
                for(ti = 0; ti < 3; ti++)
                {
                    int nntif = NFNF*ti;
                    for(tj = 0; tj < 3; tj++)
                    {
                        int nntjf = Nf*tj;
                        for(tk = 0; tk < 3; tk++)
                            val += r[newPos + nntif + nntjf + tk] * nodalWeights[ti][tj][tk];
                    }
                } // end of ti loop

                d[nnic + njc + k] = val;

            } // end of k loop
        } // end of j loop
    } // end of i loop


} // restrictResidual

void prolongateAndCorrectError(const double* __restrict__ ec, const int Nc, double* __restrict__ ef, const int Nf)
{
    int i, j, k;
    const int NCNC = Nc*Nc;
    const int NFNF = Nf*Nf;

    for(i = 0; i < Nf; i++)
    {
        const int nnif = i*NFNF;
        for(j = 0; j < Nf; j++)
        {
            const int njf = j*Nf;
            for(k = 0; k < Nf; k++)
            {
                int isNotOnCoarseEdge[3] = {i%2, j%2, k%2};
                const int val = isNotOnCoarseEdge[0] + isNotOnCoarseEdge[1] + isNotOnCoarseEdge[2];

                double retVal = 0.;
                int p;

                // if all are 1
                // => none of them are on coarse edge - so it is on center of coarse cube
                if(val == 3)
                {
                    int lowCoarseCorner[3] = {(i-1)/2, (j-1)/2, (k-1)/2};

                    // form the relevant coarse cube corners
                    int relevantCorners[8][3] = {
                        // 4 corners on X-Face
                        {lowCoarseCorner[0],   lowCoarseCorner[1],   lowCoarseCorner[2]},
                        {lowCoarseCorner[0],   lowCoarseCorner[1],   lowCoarseCorner[2]+1},
                        {lowCoarseCorner[0],   lowCoarseCorner[1]+1, lowCoarseCorner[2]},
                        {lowCoarseCorner[0],   lowCoarseCorner[1]+1, lowCoarseCorner[2]+1},

                        // 4 other corners on i+1
                        {lowCoarseCorner[0]+1, lowCoarseCorner[1],   lowCoarseCorner[2]},
                        {lowCoarseCorner[0]+1, lowCoarseCorner[1],   lowCoarseCorner[2]+1},
                        {lowCoarseCorner[0]+1, lowCoarseCorner[1]+1, lowCoarseCorner[2]},
                        {lowCoarseCorner[0]+1, lowCoarseCorner[1]+1, lowCoarseCorner[2]+1},
                    };

                    // average of all eight corners
                    for(p = 0; p < 8; p++)
                    {
                        int pos = NCNC*relevantCorners[p][0] + Nc*relevantCorners[p][1] + relevantCorners[p][2];
                        retVal += ec[pos];
                    }
                    retVal *= 0.125;
                } // end of if val==3 check

                // it is not on the coarse edge for two of the axes
                // i.e. it is on a coarse face?
                else if(val == 2)
                {
                    int coarseFaceCorners[4][3] = {};

                    // so simply check which face it is on and choose corners accordingly
                    // if on X-Face
                    if(isNotOnCoarseEdge[0] == 0)
                    {
                        int lowCornerFace[3] = {i/2, (j-1)/2, (k-1)/2};

                        // then fill in the corners
                        coarseFaceCorners[0][0] = lowCornerFace[0];   coarseFaceCorners[0][1] = lowCornerFace[1];       coarseFaceCorners[0][2] = lowCornerFace[2];
                        coarseFaceCorners[1][0] = lowCornerFace[0];   coarseFaceCorners[1][1] = lowCornerFace[1]+1;     coarseFaceCorners[1][2] = lowCornerFace[2];
                        coarseFaceCorners[2][0] = lowCornerFace[0];   coarseFaceCorners[2][1] = lowCornerFace[1];       coarseFaceCorners[2][2] = lowCornerFace[2]+1;
                        coarseFaceCorners[3][0] = lowCornerFace[0];   coarseFaceCorners[3][1] = lowCornerFace[1]+1;     coarseFaceCorners[3][2] = lowCornerFace[2]+1;
                    }

                    else if (isNotOnCoarseEdge[1] == 0)
                    {
                        int lowCornerFace[3] = {(i-1)/2, j/2, (k-1)/2};

                        // then fill in the corners
                        coarseFaceCorners[0][0] = lowCornerFace[0];   coarseFaceCorners[0][1] = lowCornerFace[1];     coarseFaceCorners[0][2] = lowCornerFace[2];
                        coarseFaceCorners[1][0] = lowCornerFace[0]+1; coarseFaceCorners[1][1] = lowCornerFace[1];     coarseFaceCorners[1][2] = lowCornerFace[2];
                        coarseFaceCorners[2][0] = lowCornerFace[0];   coarseFaceCorners[2][1] = lowCornerFace[1];     coarseFaceCorners[2][2] = lowCornerFace[2]+1;
                        coarseFaceCorners[3][0] = lowCornerFace[0]+1; coarseFaceCorners[3][1] = lowCornerFace[1];     coarseFaceCorners[3][2] = lowCornerFace[2]+1;
                    }
                    else
                    {
                        int lowCornerFace[3] = {(i-1)/2, (j-1)/2, k/2};

                        // then fill in the corners
                        coarseFaceCorners[0][0] = lowCornerFace[0];   coarseFaceCorners[0][1] = lowCornerFace[1];     coarseFaceCorners[0][2] = lowCornerFace[2];
                        coarseFaceCorners[1][0] = lowCornerFace[0];   coarseFaceCorners[1][1] = lowCornerFace[1]+1;   coarseFaceCorners[1][2] = lowCornerFace[2];
                        coarseFaceCorners[2][0] = lowCornerFace[0]+1; coarseFaceCorners[2][1] = lowCornerFace[1];     coarseFaceCorners[2][2] = lowCornerFace[2];
                        coarseFaceCorners[3][0] = lowCornerFace[0]+1; coarseFaceCorners[3][1] = lowCornerFace[1]+1;   coarseFaceCorners[3][2] = lowCornerFace[2];
                    }

                    for(p = 0; p < 4; p++)
                    {
                        int pos = NCNC*coarseFaceCorners[p][0] + Nc*coarseFaceCorners[p][1] + coarseFaceCorners[p][2];
                        retVal += ec[pos];
                    }
                    retVal *= 0.25;
                } // end of if val == 2 check

                // it is not on the edge for only one axes
                // so it is on the edge connecting two axes planes
                else if (val == 1)
                {
                    int coarseEdgeCorners[2][3] = {};

                    // check for 1 instead of 0
                    if(isNotOnCoarseEdge[0] == 1)
                    {
                        int lowCornerEdge[3] = {(i-1)/2, j/2, k/2};

                        coarseEdgeCorners[0][0] = lowCornerEdge[0];   coarseEdgeCorners[0][1] = lowCornerEdge[1];   coarseEdgeCorners[0][2] = lowCornerEdge[2];
                        coarseEdgeCorners[1][0] = lowCornerEdge[0]+1; coarseEdgeCorners[1][1] = lowCornerEdge[1];   coarseEdgeCorners[1][2] = lowCornerEdge[2];
                    }
                    else if(isNotOnCoarseEdge[1] == 1)
                    {
                        int lowCornerEdge[3] = {i/2, (j-1)/2, k/2};

                        coarseEdgeCorners[0][0] = lowCornerEdge[0];   coarseEdgeCorners[0][1] = lowCornerEdge[1];   coarseEdgeCorners[0][2] = lowCornerEdge[2];
                        coarseEdgeCorners[1][0] = lowCornerEdge[0];   coarseEdgeCorners[1][1] = lowCornerEdge[1]+1; coarseEdgeCorners[1][2] = lowCornerEdge[2];
                    }
                    else
                    {
                        int lowCornerEdge[3] = {i/2, j/2, (k-1)/2};

                        coarseEdgeCorners[0][0] = lowCornerEdge[0];   coarseEdgeCorners[0][1] = lowCornerEdge[1];   coarseEdgeCorners[0][2] = lowCornerEdge[2];
                        coarseEdgeCorners[1][0] = lowCornerEdge[0];   coarseEdgeCorners[1][1] = lowCornerEdge[1];   coarseEdgeCorners[1][2] = lowCornerEdge[2]+1;
                    }

                    for(p = 0; p < 2; p++)
                    {
                        int pos = NCNC*coarseEdgeCorners[p][0] + Nc*coarseEdgeCorners[p][1] + coarseEdgeCorners[p][2];
                        retVal += ec[pos];
                    }
                    retVal *= 0.5;
                }

                // it exactly matches the coarse grid point
                else
                    retVal = ec[NCNC*(i/2) + Nc*(j/2) + (k/2)];

                // finally update the fine grid nodal point
                ef[nnif + njf + k] += retVal;
            }
        } // end of j loop
    } // end of i loop
}

// TODO: Improve this to take in a user function pointer?
void setupBoundaryConditions(double **u, int levelN, double spacing, int numLevel)
{
    int i, j, k;
    int nni, nj;

    double *v = u[numLevel];
    //double center[2] = {GRID_LENGTH/2., GRID_LENGTH/2.};

    /***********************************/
    // X = 0 and END faces
    i = 0;
    nni = levelN*levelN*i;
    for(j = 0; j < levelN; j++)
    {
        nj = levelN*j;
        //double ty = j*spacing-center[0];
        for(k = 0; k < levelN; k++)
        {
            //double tz = k*spacing-center[1];
            //double rr = ty*ty + tz*tz;
            //if(rr <= CAPILLARY_RADIUS*CAPILLARY_RADIUS)
            v[nni + nj + k] = BCFunc(i*spacing, j*spacing, k*spacing);
        }
    }

    i = levelN-1;
    nni = levelN*levelN*i;
    for(j = 0; j < levelN; j++)
    {
        nj = levelN*j;
        //double ty = j*spacing-center[0];
        for(k = 0; k < levelN; k++)
        {
            //double tz = k*spacing-center[1];
            //double rr = ty*ty + tz*tz;

            //if(rr > (EXTRACTOR_INNER_RADIUS*EXTRACTOR_INNER_RADIUS)
            //        &&
            //   rr < (EXTRACTOR_OUTER_RADIUS*EXTRACTOR_OUTER_RADIUS))
            v[nni + nj + k] = BCFunc(i*spacing, j*spacing, k*spacing);
        }
    }
    /***********************************/
    /***********************************/
    // Y = 0 and END faces
    j = 0;
    nj = levelN * j;
    for(i = 0; i < levelN; i++)
    {
        nni = levelN*levelN*i;
        for(k = 0; k < levelN; k++)
            v[nni + nj + k] = BCFunc(i*spacing, j*spacing, k*spacing);
    }

    j = levelN-1;
    nj = levelN * j;
    for(i = 0; i < levelN; i++)
    {
        nni = levelN*levelN*i;
        for(k = 0; k < levelN; k++)
            v[nni + nj + k] = BCFunc(i*spacing, j*spacing, k*spacing);
    }
    /***********************************/
    /***********************************/
    // Z = 0 and END faces
    k = 0;
    for(i = 0; i < levelN; i++)
    {
        nni = levelN*levelN*i;
        for(j = 0; j < levelN; j++)
        {
            nj = levelN*j;
            v[nni + nj + k] = BCFunc(i*spacing, j*spacing, k*spacing);
        }

    }

    k = levelN-1;
    for(i = 0; i < levelN; i++)
    {
        nni = levelN*levelN*i;
        for(j = 0; j < levelN; j++)
        {
            nj = levelN*j;
            v[nni + nj + k] = BCFunc(i*spacing, j*spacing, k*spacing);
        }
    }
    /***********************************/
    /***********************************/
} // end of setupBoundaryConditions

// return current l2-norm squared of residual
double vcycle(double **u, double **f, int q, const int numLevels, const int smootherIter, int N, double *LU)
{
    double h = GRID_LENGTH/(N-1);

    double *v = u[q];
    double *d = f[q];

    // reset lower level soln to zero
    if(q < (numLevels-1))
        memset(v, 0, N*N*N*sizeof(double));

    double timingTemp;
    if(q == 0)
    {
        timingTemp = clock();
        // prepare A matrix to send for gaussian elimination

        const int NN = N*N;
        const int totalNodes = NN*N;
        solveWithLU(LU, totalNodes, f[q], u[q]);
        tInfo[q][3].timeTaken += (clock() - timingTemp);
        tInfo[q][3].numCalls++;

        return 0.;
    }

    timingTemp = clock();
    GaussSeidelSmoother(v, d, N, h, smootherIter);
    tInfo[q][0].timeTaken += (clock() - timingTemp);
    tInfo[q][0].numCalls++;

    // allocate the residual vector
    double *r = calloc(N*N*N, sizeof(double));

    timingTemp = clock();
    calculateResidual(v, d, N, h, r);
    tInfo[q][1].timeTaken += (clock() - timingTemp);
    tInfo[q][1].numCalls++;

    // update N for next coarser level
    int N_coarse = (N+1)/2;

    // now restrict this onto the next level
    double *d1 = f[q-1];

    timingTemp = clock();
    restrictResidual(r, N, d1, N_coarse);
    tInfo[q][2].timeTaken += (clock() - timingTemp);
    tInfo[q][2].numCalls++;
    // free the residual memory used
    free(r);

    // do recursive call now
    timingTemp = clock();
    vcycle(u, f, q-1, numLevels, smootherIter, N_coarse, LU);
    tInfo[q][3].timeTaken += (clock() - timingTemp);
    tInfo[q][3].numCalls++;

    // now do prolongation to the fine grid
    double *v1 = u[q-1];
    // PARALLELIZABLE
    timingTemp = clock();
    prolongateAndCorrectError(v1, N_coarse, v, N);
    tInfo[q][4].timeTaken += (clock() - timingTemp);
    tInfo[q][4].numCalls++;

    timingTemp = clock();
    GaussSeidelSmoother(v, d, N, h, smootherIter);
    tInfo[q][5].timeTaken += (clock() - timingTemp);
    tInfo[q][5].numCalls++;

    timingTemp = clock();
    double res = calculateResidual(v, d, N, h, NULL);
    tInfo[q][6].timeTaken += (clock() - timingTemp);
    tInfo[q][6].numCalls++;

    return res;
}

void SolverFMGInitialize()
{
    // solve on the coarsest grid
    int N = coarseGridNum;
    const int NN = N*N;
    const int totalNodes = NN*N;

    // impose BCs on the coarsest level
    double h = GRID_LENGTH/(coarseGridNum-1);
    setupBoundaryConditions(u, N, h, 0);

    // A should now be storing its LU counterpart
    solveWithLU(A, totalNodes, d[0], u[0]);

    // for loop
    int l, Nc;
    for(l = 1; l < numLevels; l++)
    {
        Nc = N;         // previous coarse num count
        N = 2*N-1;      // finer grid count
        h = h*0.5;      // halved spacing

        // Interpolate previous level soln
        // IMPORTANT: u[l] must have been ZEROED out at this point
        prolongateAndCorrectError(u[l-1], Nc, u[l], N);

        // setupBoundaryConditions on this level
        setupBoundaryConditions(u, N, h, l);

        // set previous level soln to zero?
        memset(u[l-1], 0, sizeof(double)*Nc*Nc*Nc);

        // do vcycles
        vcycle(u, d, l, numLevels, gsIterNum, N, A);
    }
}

void updateEdgeValues(double *u, const int N)
{
    const int NN = N*N;
    int i, j, k;
    int pos;

    // update the 8 corner point values first
    // 4 points on X = 0 face
    i = 0;
    j = 0; k = 0;
    pos = NN*i + N*j + k;
    u[pos] = (1./3) * (u[pos+1] + u[pos+N] + u[pos+NN]);

    j = 0; k = N-1;
    pos = NN*i + N*j + k;
    u[pos] = (1./3) * (u[pos-1] + u[pos+N] + u[pos+NN]);

    j = N-1; k = 0;
    pos = NN*i + N*j + k;
    u[pos] = (1./3) * (u[pos+1] + u[pos-N] + u[pos+NN]);

    j = N-1; k = N-1;
    pos = NN*i + N*j + k;
    u[pos] = (1./3) * (u[pos-1] + u[pos-N] + u[pos+NN]);

    // 4 points on X=N-1 face
    i = N-1;
    j = 0; k = 0;
    pos = NN*i + N*j + k;
    u[pos] = (1./3) * (u[pos+1] + u[pos+N] + u[pos-NN]);

    j = 0; k = N-1;
    pos = NN*i + N*j + k;
    u[pos] = (1./3) * (u[pos-1] + u[pos+N] + u[pos-NN]);

    j = N-1; k = 0;
    pos = NN*i + N*j + k;
    u[pos] = (1./3) * (u[pos+1] + u[pos-N] + u[pos-NN]);

    j = N-1; k = N-1;
    pos = NN*i + N*j + k;
    u[pos] = (1./3) * (u[pos+1] + u[pos-N] + u[pos-NN]);


    // update the 12 edges
    // X = 0 face
    i = 0; k = 0;
    for(j = 1; j < N-1; j++)
    {
        pos = NN*i + N*j + k;
        u[pos] = 0.5 * (u[pos+1] + u[pos+NN]);
    }

    i = 0; k = N-1;
    for(j = 1; j < N-1; j++)
    {
        pos = NN*i + N*j + k;
        u[pos] = 0.5 * (u[pos-1] + u[pos+NN]);
    }

    i = 0; j = 0;
    for(k = 1; k < N-1; k++)
    {
        pos = NN*i + N*j + k;
        u[pos] = 0.5 * (u[pos+N] + u[pos+NN]);
    }
    i = 0; j = N-1;
    for(k = 1; k < N-1; k++)
    {
        pos = NN*i + N*j + k;
        u[pos] = 0.5 * (u[pos-N] + u[pos+NN]);
    }

    // X = N-1 face
    i = N-1; k = 0;
    for(j = 1; j < N-1; j++)
    {
        pos = NN*i + N*j + k;
        u[pos] = 0.5 * (u[pos+1] + u[pos-NN]);
    }

    i = N-1; k = N-1;
    for(j = 1; j < N-1; j++)
    {
        pos = NN*i + N*j + k;
        u[pos] = 0.5 * (u[pos-1] + u[pos-NN]);
    }

    i = N-1; j = 0;
    for(k = 1; k < N-1; k++)
    {
        pos = NN*i + N*j + k;
        u[pos] = 0.5 * (u[pos+N] + u[pos-NN]);
    }
    i = N-1; j = N-1;
    for(k = 1; k < N-1; k++)
    {
        pos = NN*i + N*j + k;
        u[pos] = 0.5 * (u[pos-N] + u[pos-NN]);
    }

    // Y = 0 face
    j = 0; k = 0;
    for(i = 1; i < N-1; i++)
    {
        pos = NN*i + N*j + k;
        u[pos] = 0.5 * (u[pos+N] + u[pos+1]);
    }
    j = 0; k = N-1;
    for(i = 1; i < N-1; i++)
    {
        pos = NN*i + N*j + k;
        u[pos] = 0.5 * (u[pos+N] + u[pos-1]);
    }
    // Y = N-1 face
    j = N-1; k = 0;
    for(i = 1; i < N-1; i++)
    {
        pos = NN*i + N*j + k;
        u[pos] = 0.5 * (u[pos-N] + u[pos+1]);
    }
    j = N-1; k = N-1;
    for(i = 1; i < N-1; i++)
    {
        pos = NN*i + N*j + k;
        u[pos] = 0.5 * (u[pos-N] + u[pos-1]);
    }
}


/*
 * Understand storage and strided access patterns
 * And areas for parallelism
 */

void SolverSetupBoundaryConditions()
{ return setupBoundaryConditions(u, finestOneSideNum, spacing, numLevels-1); }

double SolverLinSolve()
{
    double res = vcycle(u, d, numLevels-1, numLevels, gsIterNum, finestOneSideNum, A);
    return res;
}

void SolverSmoothenEdgeValues()
{ return updateEdgeValues(u[numLevels-1], finestOneSideNum); }

double SolverGetResidual()
{
    return calculateResidual(u[numLevels-1], d[numLevels-1], finestOneSideNum, spacing, NULL);
}

void SolverResetTimingInfo()
{ resetTimingInfo(tInfo, numLevels); }

void SolverPrintTimingInfo()
{ printTimingInfo(tInfo, numLevels); }

int main(int argc, char** argv)
{
    // parse the passed in options
    SolverInitialize(argc, argv);

    // enforce the boundary conditions
    setupBoundaryConditions(u, finestOneSideNum, spacing, numLevels-1);

    double norm = 1e9, tolerance = 1e-8;

    // compare against squared tolerance, can avoid
    // unnecessary sqrts that way?
    // RELATIVE CONVERGENCE criteria
    const double initResidual = calculateResidual(u[numLevels-1], d[numLevels-1], finestOneSideNum, spacing, NULL);

    const double cmpNorm = initResidual*tolerance;
    int iterCount = 1;

    if(useFMG)
    {
        printf("Doing FMG Initialization....");
        SolverFMGInitialize();
        printf("done\n");
    }

    double relResidualRatio = -1;
    double oldNorm = -1;
    while(norm > cmpNorm)
    {
        oldNorm = norm;
        norm = vcycle(u, d, numLevels-1, numLevels, gsIterNum, finestOneSideNum, A);
        relResidualRatio = norm/oldNorm;
        //norm = calculateResidual(u[numLevels-1], d[numLevels-1], finestOneSideNum, h);
        printf("%5d    Residual Norm:%20g     ResidRatio:%20g\n", iterCount, norm, relResidualRatio);
        iterCount++;
    }

    // smoothen border edge and point values
    // although they are not used in the calculation
    //updateEdgeValues(u[numLevels-1], finestOneSideNum);

    printTimingInfo(tInfo, numLevels);

    // checking against analytical soln
    double errNorm = 0.;
    int i, j, k;
    double *v = u[numLevels-1];
    for(i = 0; i < finestOneSideNum; i++)
    {
        const int nni = finestOneSideNum*finestOneSideNum*i;
        for(j = 0; j < finestOneSideNum; j++)
        {
            const int nj = finestOneSideNum * j;
            for(k = 0; k < finestOneSideNum; k++)
            {
                const int pos = nni + nj + k;
                double diff = v[pos] - BCFunc(i*spacing, j*spacing, k*spacing);
                v[pos] = diff;
                errNorm += diff*diff;
            }
        }
    }
    errNorm = sqrt(errNorm);
    printf("Error norm: %10.6g\n", errNorm);

    writeOutputData("diff_ref.vtk", v, spacing, finestOneSideNum);

    deAllocGridLevels(&d, numLevels);
    deAllocGridLevels(&u, numLevels);

    deAllocTimingInfo(&tInfo, numLevels);
    free(A);

    return 0;
}

