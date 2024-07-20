#include <cuda_runtime.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>
#include <vector>
#include <atomic>
#include "particle.cuh"
#include "computehelper.cuh"
#include "cuda_helper.cuh"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "kernel.cuh"

const std::string inputFilename = "../scenes/scene_2.vtk";
const int nParticles = 480470;
const int nParticlesFluid = 65220;
const double dt = 0.000625;
const double boxSize = 3.0;
const double h = 0.045;
const int nSteps = 8000;
const int vtkOutputFrequency = 100;
const double gravity = 9.81;
const double restDensity = 1000.0;
const double viscosityCoefficient = 50.0;
const double gasConstant = 20.0;

// function to read particles data from a VTK file
void readVTK(const std::string &filename, int numParticles, thrust::host_vector<Particle> &particles)
{
    std::string line;
    int numPoints;

    // open file for reading
    std::ifstream file(inputFilename);
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
    file >> line >> numPoints >> line; // POINTS numPoints float
    if (numPoints != numParticles)
    {
        std::cerr << "Number of points in file does not match the provided number of particles." << std::endl;
        return;
    }
    for (int i = 0; i < numParticles; ++i)
    {
        file >> particles[i].pos.x >> particles[i].pos.y >> particles[i].pos.z;
        // std::cout << particles[i].x << std::endl;
    }

    // skip to the POINT_DATA section
    while (std::getline(file, line) && line.substr(0, 10) != "POINT_DATA")
        ;

    // masses
    file >> line >> line >> line; // SCALARS mass double
    file >> line >> line;         // LOOKUP_TABLE default
    for (int i = 0; i < numParticles; ++i)
    {
        file >> particles[i].mass;
        // std::cout << "mass " << particles[i].mass << std::endl;
    }

    // velocities
    file >> line >> line >> line; // VECTORS velocity double
    for (int i = 0; i < numParticles; ++i)
    {
        file >> particles[i].vel.x >> particles[i].vel.y >> particles[i].vel.z;
        // std::cout << "velocity x " << particles[i].vx <<std::endl;
    }

    // radii
    file >> line >> line >> line; // SCALARS radius double
    file >> line >> line;         // LOOKUP_TABLE default
    for (int i = 0; i < numParticles; ++i)
    {
        file >> particles[i].radius;
        // std::cout << "radius " << particles[i].radius << std::endl;
    }

    // fix
    file >> line >> line >> line; // SCALARS radius double
    file >> line >> line;         // LOOKUP_TABLE default
    for (int i = 0; i < numParticles; ++i)
    {
        file >> particles[i].ghost;
        // std::cout << "fix " << particles[i].ghost << std::endl;
    }

    // close file
    file.close();
}

// function to write particles data to a VTK file
void writeVTK(thrust::host_vector<Particle> &particles, int step, int nParticles)
{
    // open file for writing
    std::ostringstream filename;
    filename << "./output/" << "particles_" << std::setw(4) << std::setfill('0') << step << ".vtk";
    std::ofstream file(filename.str());

    // header and dataset
    file << "# vtk DataFile Version 2.0\n";
    file << "Particle data\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";

    // points section
    file << "POINTS " << nParticlesFluid << " float\n";
    for (int i = 0; i < nParticles; i++)
    {
        if (particles[i].ghost == false)
        {
            file << particles[i].pos.x << " " << particles[i].pos.y << " " << particles[i].pos.z << "\n";
        }
    }

    // cells section
    file << "CELLS 0 0\n";
    file << "CELL_TYPES 0\n";

    // point data section
    file << "POINT_DATA " << nParticlesFluid << "\n";

    // mass
    file << "SCALARS m float\n";
    file << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nParticles; i++)
    {
        if (particles[i].ghost == false)
        {
            file << particles[i].mass << "\n";
        }
    }

    // density
    file << "SCALARS rho float\n";
    file << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nParticles; i++)
    {
        if (particles[i].ghost == false)
        {
            file << particles[i].density << "\n";
        }
    }

    // radii
    file << "SCALARS r float\n";
    file << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nParticles; i++)
    {
        if (particles[i].ghost == false)
        {
            file << particles[i].radius << "\n";
        }
    }

    // positions
    file << "VECTORS p float\n";
    for (int i = 0; i < nParticles; i++)
    {
        if (particles[i].ghost == false)
        {
            file << particles[i].pos.x << " " << particles[i].pos.y << " " << particles[i].pos.z << "\n";
        }
    }

    // velocities
    file << "VECTORS v float\n";
    for (int i = 0; i < nParticles; i++)
    {
        if (particles[i].ghost == false)
        {
            file << particles[i].vel.x << " " << particles[i].vel.y << " " << particles[i].vel.z << "\n";
        }
    }
}

int main()
{
    kernel k(h);
    kernel *d_k;
    cudaMalloc(&d_k, sizeof(kernel));
    cudaMemcpy(d_k, &k, sizeof(kernel), cudaMemcpyHostToDevice);

    // allocate host memory
    thrust::host_vector<Particle> particles(nParticles);
    
    readVTK(inputFilename, nParticles, particles);
    int nCellsPerSide = static_cast<int>(boxSize / h);
    int nCells = nCellsPerSide * nCellsPerSide * nCellsPerSide;
    double cellSize = boxSize / nCellsPerSide;

    // allocate device memory
    thrust::device_vector<Particle> d_particles_vec = particles;
    thrust::device_vector<int> CellDS_vec(nCells);
    thrust::device_vector<int> PartDS_vec(nParticles);
    Particle *d_particles = thrust::raw_pointer_cast(d_particles_vec.data());
    int *d_cell_ptr = thrust::raw_pointer_cast(CellDS_vec.data());
    int *d_part_ptr = thrust::raw_pointer_cast(PartDS_vec.data());
    
    auto [thPerBlk, blks] = setgpuconfig();

    for (int step = 0; step < nSteps; ++step)
    {
        // initialize cells on device
        setCellDS<<<blks, thPerBlk>>>(d_cell_ptr, nCells);
        checkCuda(cudaGetLastError(), __LINE__ - 1);
        setPartDS<<<blks, thPerBlk>>>(d_part_ptr, nParticles);
        checkCuda(cudaGetLastError(), __LINE__ - 1);

        // build linked lists on device
        updateDS<<<blks, thPerBlk>>>(d_particles, d_cell_ptr, d_part_ptr, cellSize, nCellsPerSide, nParticles);
        checkCuda(cudaGetLastError(), __LINE__ - 1);

        // compute densities
        updatePressureDensity<<<blks, thPerBlk>>>(d_particles, d_cell_ptr, d_part_ptr, cellSize, nCellsPerSide, nParticles, h,restDensity, gasConstant, d_k);
        checkCuda(cudaGetLastError(), __LINE__ - 1);

        // calculate forces
        updateAcceleration<<<blks, thPerBlk>>>(d_particles, d_cell_ptr, d_part_ptr, cellSize, nCellsPerSide, h, nParticles, viscosityCoefficient, gravity, d_k);
        checkCuda(cudaGetLastError(), __LINE__ - 1);

        // symplectic Euler Integration
        updateState<<<blks, thPerBlk>>>(d_particles, dt, nParticles);
        checkCuda(cudaGetLastError(), __LINE__ - 1);

        // copy particles back to host for VTK output
        if (step % vtkOutputFrequency == 0)
        {
            // checkCuda(cudaMemcpy(particles, d_particles, nParticles * sizeof(Particle), cudaMemcpyDeviceToHost), 12);
            particles = d_particles_vec;
            // writeVTK(particles, step, nParticles);
            writeVTK(particles, step, nParticles);
            std::cout << "Step " << step << " done." << std::endl;
        }
    }
    return 0;
}