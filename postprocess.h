#ifndef POSTPROCESS_H
#define POSTPROCESS_H

// function for writing out values
void writeOutputData(const char* fileName, const double *grid, const double h, const int N)
{
    FILE* fileValues = fopen(fileName, "w");
    int i, j, k;

    const int totalNodes = N*N*N;

    // write the VTK header
    fprintf(fileValues, "# vtk DataFile Version 2.0\n"
                        "Potential data\n"
                        "ASCII\n"
                        "DATASET STRUCTURED_GRID\n"
                        "DIMENSIONS %d %d %d\n"
                        "POINTS %d float\n", N, N, N, totalNodes
                        );

    // write out the point locations
    for(i = 0; i < N; i++)
    {
        double x = h * i;
        for(j = 0; j < N; j++)
        {
            double y = h * j;
            for(k = 0; k < N; k++)
            {
                double z = h * k;
                fprintf(fileValues, "%10.8e %10.8e %10.8e\n", x, y, z);
            }
        }
    }

    // now write out the potential values
    fprintf(fileValues, "\n"
                        "POINT_DATA %d\n"
                        "SCALARS data float 1\n"
                        "LOOKUP_TABLE default\n", totalNodes
            );
    int count = 0;
    for(count = 0; count < totalNodes; count++)
        fprintf(fileValues, "%10.8e\n", grid[count]);

    fclose(fileValues);
}

#endif
