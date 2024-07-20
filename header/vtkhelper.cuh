#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "thrust/host_vector.h"
#include "particle.cuh"


// function to read particles data from a VTK file
void readVTK(const std::string &filename, int numParticles, thrust::host_vector<Particle> &particles)
{
    std::string line;
    int numPoints;

    // open file for reading
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // skip header lines
    for (int i = 0; i < 4; ++i)
    {
        std::getline(file, line);
    }

    // points (positions)
    file >> line >> numPoints >> line; // POINTS numPoints double
    if (numPoints != numParticles)
    {
        std::cerr << "Number of points in file does not match the provided number of particles." << std::endl;
        return;
    }
    for (int i = 0; i < numParticles; ++i)
    {
        file >> particles[i].pos.x >> particles[i].pos.y >> particles[i].pos.z;
    }

    // skip to the POINT_DATA section
    while (std::getline(file, line) && line.substr(0, 10) != "POINT_DATA");

    // masses
    file >> line >> line >> line; // SCALARS mass double
    file >> line >> line;         // LOOKUP_TABLE default
    for (int i = 0; i < numParticles; ++i)
    {
        file >> particles[i].mass;
    }


    file >> line >> line >> line; // VECTORS velocity double
    for (int i = 0; i < numParticles; ++i)
    {
        file >> particles[i].vel.x >> particles[i].vel.y >> particles[i].vel.z;
    }


    file >> line >> line >> line; // SCALARS radius double
    file >> line >> line;         // LOOKUP_TABLE default
    for (int i = 0; i < numParticles; ++i)
    {
        file >> line;
    }

    file >> line >> line >> line; // SCALARS radius double
    file >> line >> line;         // LOOKUP_TABLE default
    for (int i = 0; i < numParticles; ++i)
    {
        file >> particles[i].ghost;
    }

    file.close();
}

// function to write particles data to a VTK file
void writeVTK(thrust::host_vector<Particle> &particles, int step, int nParticlesFluid)
{
    // open file for writing
    std::ostringstream filename;
    filename << "./output/" << "particles_" << std::setw(4) << std::setfill('0') << step << ".vtk";
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