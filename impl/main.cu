
#include "particle.cuh"
#include "computehelper.cuh"
#include "cuda_helper.cuh"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "kernel.cuh"
#include "vtkhelper.cuh"

const std::string inputFilename = "../scenes/sph_scene_setup.vtk";
const int nParticles = 1316944;
const int nParticlesFluid = 2800;
const double dt = 0.000625;
const double boxSize = 3.0;
const double h = 0.065;
const int nSteps = 8000;
const int vtkOutputFrequency = 100;
const double gravity = 9.81;
const double restDensity = 1000.0;
const double viscosityCoefficient = 50.0;
const double gasConstant = 20.0;


int main()
{
    auto [thPerBlk, blks] = setgpuconfig();

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

    thrust::device_vector<Particle> d_particles_vec = particles;
    thrust::device_vector<int> CellDS_vec(nCells);
    thrust::device_vector<int> PartDS_vec(nParticles);
    Particle *d_particles = thrust::raw_pointer_cast(d_particles_vec.data());
    int *d_cell_ptr = thrust::raw_pointer_cast(CellDS_vec.data());
    int *d_part_ptr = thrust::raw_pointer_cast(PartDS_vec.data());

    for (int step = 0; step < nSteps; ++step)
    {
        setDS<<<blks, thPerBlk>>>(d_cell_ptr, nCells);
        checkCuda(cudaGetLastError(), __LINE__ - 1);
        setDS<<<blks, thPerBlk>>>(d_part_ptr, nParticles);
        checkCuda(cudaGetLastError(), __LINE__ - 1);

        updateDS<<<blks, thPerBlk>>>(d_particles, d_cell_ptr, d_part_ptr, cellSize, nCellsPerSide, nParticles);
        checkCuda(cudaGetLastError(), __LINE__ - 1);

        updatePressureDensity<<<blks, thPerBlk>>>(d_particles, d_cell_ptr, d_part_ptr, cellSize, nCellsPerSide, nParticles, h, restDensity, gasConstant, d_k);
        checkCuda(cudaGetLastError(), __LINE__ - 1);

        updateAcceleration<<<blks, thPerBlk>>>(d_particles, d_cell_ptr, d_part_ptr, cellSize, nCellsPerSide, h, nParticles, viscosityCoefficient, gravity, d_k);
        checkCuda(cudaGetLastError(), __LINE__ - 1);

        updateState<<<blks, thPerBlk>>>(d_particles, dt, nParticles);
        checkCuda(cudaGetLastError(), __LINE__ - 1);

        // copy particles back to host for VTK output
        if (step % vtkOutputFrequency == 0)
        {
            particles = d_particles_vec;
            writeVTK(particles, step, nParticlesFluid);
            std::cout << "Step " << step << " done." << std::endl;
        }
    }
    return 0;
}