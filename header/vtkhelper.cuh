#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "thrust/host_vector.h"
#include "particle.cuh"


// function to write particles data to a VTK file
void writeVTK(thrust::host_vector<Particle> &particles, int step, int nParticlesFluid)
{
    // open file for writing
    std::ostringstream filename;
    filename << "./output/" << "output_" <<  step << ".vtk";
    std::ofstream file(filename.str());

    // header and dataset
    file << "# vtk DataFile Version 4.0 \n";
    file << "hesp visualization file \n";
    file << "ASCII \n";
    file << "DATASET UNSTRUCTURED_GRID\n";

    // points section
    file << "POINTS " << nParticlesFluid << " double\n";
    for (const auto &particle : particles)
    {
        if (!particle.ghost)
        {
            file << particle.pos.x << " " << particle.pos.y << " " << particle.pos.z << "\n";
        }
    }

    file << "CELLS 0 0\n";
    file << "CELL_TYPES 0\n";
    file << "POINT_DATA " << nParticlesFluid << "\n";
    file << "SCALARS density double\n";
    file << "LOOKUP_TABLE default\n";

    for (const auto &particle : particles)
    {
        if (!particle.ghost)
        {
            file << std::fixed << std::setprecision(5) << particle.density << std::endl;
        }
    }

    file << "VECTORS velocity double" << "\n";

    for (const auto &particle : particles)
    {
        if (!particle.ghost)
        {
            file << std::fixed << std::setprecision(5)
                 << particle.vel.x << " "
                 << particle.vel.y << " "
                 << particle.vel.z << "" << "\n";
        }
    }
}