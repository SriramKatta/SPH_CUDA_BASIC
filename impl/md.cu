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

std::string inputFilename;
int nParticles;
int nParticlesFluid;
double dt;
double boxSize;
double h;
int nSteps;
int vtkOutputFrequency;
double gravity;
double particleRadius;
double restDensity;
double viscosityCoefficient;
double gasConstant;


// function to read configuration file
void readConfig() {
    std::string filename = "configuration.config";
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string key;
        if (std::getline(iss, key, '=')) {
            key.erase(key.find_last_not_of(" \n\r\t") + 1);  // trim right
            std::string value;
            if (std::getline(iss, value)) {
                value.erase(0, value.find_first_not_of(" \n\r\t"));  // trim left
                if (key == "inputFilename") {
                    inputFilename = value;
                    inputFilename.erase(0, 1);
                    inputFilename.pop_back();
                    inputFilename.pop_back();
                } else if (key == "nParticles") {
                    nParticles = std::stoi(value);
                } else if (key == "nParticlesFluid") {
                    nParticlesFluid = std::stoi(value);
                } else if (key == "dt") {
                    dt = std::stod(value);
                } else if (key == "nSteps") {
                    nSteps = std::stoi(value);
                } else if (key == "vtkOutputFrequency") {
                    vtkOutputFrequency = std::stoi(value);
                } else if (key == "boxSize") {
                    boxSize = std::stod(value);
                } else if (key == "h") {
                    h = std::stod(value);
                } else if (key == "gravity"){
                    gravity = std::stod(value);
                } else if (key == "restDensity"){
                    restDensity = std::stod(value);
                } else if (key == "viscosityCoefficient"){
                    viscosityCoefficient = std::stod(value);
                } else if (key == "gasConstant"){
                    gasConstant = std::stod(value);
                }
            }
        }
    }
}

// function to read particles data from a VTK file
void readVTK(const std::string &filename, int numParticles, thrust::host_vector<Particle>& particles) {
    std::string line;
    int numPoints;

    // open file for reading
    std::ifstream file(inputFilename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // skip header lines
    for (int i = 0; i < 4; ++i) {
        std::getline(file, line);
    }

    // points (positions)
    file >> line >> numPoints >> line;      // POINTS numPoints float
    if (numPoints != numParticles) {
        std::cerr << "Number of points in file does not match the provided number of particles." << std::endl;
        return;
    }
    for (int i = 0; i < numParticles; ++i) {
        file >> particles[i].pos.x >> particles[i].pos.y >> particles[i].pos.z;
        //std::cout << particles[i].x << std::endl;
    }

    // skip to the POINT_DATA section
    while (std::getline(file, line) && line.substr(0, 10) != "POINT_DATA");

    // masses
    file >> line >> line >> line;           // SCALARS mass double
    file >> line >> line;                   // LOOKUP_TABLE default
    for (int i = 0; i < numParticles; ++i) {
        file >> particles[i].mass;
        //std::cout << "mass " << particles[i].mass << std::endl;
    }

    // velocities
    file >> line >> line >> line;           // VECTORS velocity double
    for (int i = 0; i < numParticles; ++i) {
        file >> particles[i].vel.x >> particles[i].vel.y >> particles[i].vel.z;
        //std::cout << "velocity x " << particles[i].vx <<std::endl;
    }

    // radii
    file >> line >> line >> line;           // SCALARS radius double
    file >> line >> line;                   // LOOKUP_TABLE default
    for (int i = 0; i < numParticles; ++i) {
        file >> particles[i].radius;
        //std::cout << "radius " << particles[i].radius << std::endl;
    }

    // fix
    file >> line >> line >> line;           // SCALARS radius double
    file >> line >> line;                   // LOOKUP_TABLE default
    for (int i = 0; i < numParticles; ++i) {
        file >> particles[i].fix;
        //std::cout << "fix " << particles[i].fix << std::endl;
    }

    // close file
    file.close();
}

// function to write particles data to a VTK file
void writeVTK(thrust::host_vector<Particle>& particles, int step, int nParticles) {
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
    for (int i = 0; i < nParticles; i++) {
        if (particles[i].fix == false) {
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
    for (int i = 0; i < nParticles; i++) {
        if (particles[i].fix == false) {
            file << particles[i].mass << "\n";
        }
    }

    // density
    file << "SCALARS rho float\n";
    file << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nParticles; i++) {
        if (particles[i].fix == false) {
            file << particles[i].density << "\n";
        }
    }

    // radii
    file << "SCALARS r float\n";
    file << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nParticles; i++) {
        if (particles[i].fix == false) {
            file << particles[i].radius << "\n";
        }
    }

    // positions
    file << "VECTORS p float\n";
    for (int i = 0; i < nParticles; i++) {
        if (particles[i].fix == false) {
            file << particles[i].pos.x << " " << particles[i].pos.y << " " << particles[i].pos.z << "\n";
        }
    }

    // velocities
    file << "VECTORS v float\n";
    for (int i = 0; i < nParticles; i++) {
        if (particles[i].fix == false) {
            file << particles[i].vel.x << " " << particles[i].vel.y << " " << particles[i].vel.z << "\n";
        }
    }
}





int main() {
    // initialize host variables I
    readConfig();

    // allocate host memory
    thrust::host_vector<Particle> particles(nParticles);
    //cudaMallocHost(&particles, nParticles * sizeof(Particle));

    // initialize host variables II
    readVTK(inputFilename, nParticles, particles);
    int nCellsPerSide = static_cast<int>(boxSize / h);
    int nCells = nCellsPerSide * nCellsPerSide * nCellsPerSide;
    double cellSize = boxSize / nCellsPerSide;

    // allocate device memory
    thrust::device_vector<Particle> d_particles_vec = particles;
    thrust::device_vector<int> d_cellHead_vec(nCells);
    thrust::device_vector<int> d_cellNext_vec(nParticles);
    Particle* d_particles = thrust::raw_pointer_cast(d_particles_vec.data());
    int* d_cellHead = thrust::raw_pointer_cast(d_cellHead_vec.data());
    int* d_cellNext = thrust::raw_pointer_cast(d_cellNext_vec.data());
    //cudaMalloc(&d_particles, nParticles * sizeof(Particle));
    //cudaMalloc(&d_cellHead, nCells * sizeof(int));
    //cudaMalloc(&d_cellNext, nParticles * sizeof(int));
//
    //// copy particle data to device
    //checkCuda(cudaMemcpy(d_particles, particles, nParticles * sizeof(Particle), cudaMemcpyHostToDevice), 11);

    // block-thread-partitioning
    int threadsPerBlock = 256;
    int blocksPerGridParticles = (nParticles + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridCells = (nCells + threadsPerBlock - 1) / threadsPerBlock;

    // time start point
    auto start = std::chrono::steady_clock::now();

    // simulation step loop
    for (int step = 0; step < nSteps; ++step) {
        // initialize cells on device
        initializeCellHeadKernel<<<blocksPerGridCells, threadsPerBlock>>>(d_cellHead, nCells);
        initializeCellNextKernel<<<blocksPerGridParticles, threadsPerBlock>>>(d_cellNext, nParticles);
        checkCuda(cudaGetLastError(), 1);

        // build linked lists on device
        buildLinkedListsKernel<<<blocksPerGridParticles, threadsPerBlock>>>(d_particles, d_cellHead, d_cellNext, cellSize, nCellsPerSide, nParticles);
        checkCuda(cudaGetLastError(), 2);

        // compute densities
        computeDensitiesKernel<<<blocksPerGridParticles, threadsPerBlock>>>(d_particles, d_cellHead, d_cellNext, cellSize, nCellsPerSide, h, nParticles, restDensity, gasConstant);
        checkCuda(cudaGetLastError(), 3);

        // calculate forces
        calculateForcesKernel<<<blocksPerGridParticles, threadsPerBlock>>>(d_particles, d_cellHead, d_cellNext, cellSize, nCellsPerSide, h, nParticles, viscosityCoefficient, gravity);
        checkCuda(cudaGetLastError(), 4);

        // symplectic Euler Integration
        updateState<<<blocksPerGridParticles, threadsPerBlock>>>(d_particles, dt, nParticles);
        checkCuda(cudaGetLastError(), 5);

        // copy particles back to host for VTK output
        if (step % vtkOutputFrequency == 0) {
            //checkCuda(cudaMemcpy(particles, d_particles, nParticles * sizeof(Particle), cudaMemcpyDeviceToHost), 12);
            particles = d_particles_vec;
            //writeVTK(particles, step, nParticles);
            writeVTK(particles, step, nParticles);
            std::cout << "Step " << step << " done." << std::endl;
        }
    }

    // time measurement
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::nano> elapsed_seconds = (end - start);
    double meanExecutionTime = elapsed_seconds.count() / nSteps / 1e6;
    std::cout << "Simulation complete." << std::endl;
    std::cout << "Mean execution time per time step: " << meanExecutionTime << " ms" << std::endl;

    // free memory
    //cudaFree(d_particles);
    //cudaFree(d_cellHead);
    //cudaFree(d_cellNext);

    return 0;
}