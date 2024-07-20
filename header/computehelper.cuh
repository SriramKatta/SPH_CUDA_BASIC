#pragma once

#include "particle.cuh"
#include "vector_helper.cuh"

using vec3d = double3;
using vec3i = int3;

// function to get cell index in x,y,z
__device__ vec3i getCellIndex(const vec3d &pos, const double &cellSize, const int &nCellsPerSide)
{
  vec3i cellIndex = make_int3(
      static_cast<int>(pos.x / cellSize + nCellsPerSide) % nCellsPerSide,
      static_cast<int>(pos.y / cellSize + nCellsPerSide) % nCellsPerSide,
      static_cast<int>(pos.z / cellSize + nCellsPerSide) % nCellsPerSide);
  return cellIndex;
}

__device__ int getCellIndex1D(const vec3i &cellIndex, int nCellsPerSide)
{
  return cellIndex.x * nCellsPerSide * nCellsPerSide + cellIndex.y * nCellsPerSide + cellIndex.z;
  ;
}

// CUDA kernel to initialize cellHead
__global__ void initializeCellHeadKernel(int *d_cellHead, int nCells)
{
  GRID_STRIDE(index, stride);
  for (int i = index; i < nCells; i += stride)
  {
    d_cellHead[i] = -1;
  }
}

// CUDA kernel to initialize cellNext
__global__ void initializeCellNextKernel(int *d_cellNext, int nParticles)
{
  GRID_STRIDE(index, stride);
  for (int i = index; i < nParticles; i += stride)
  {
    d_cellNext[i] = -1;
  }
}

// CUDA kernel to build linked lists
__global__ void buildLinkedListsKernel(Particle *d_particles,
                                       int *CellDs,
                                       int *PartDS,
                                       double cellSize,
                                       int nCellsPerSide,
                                       int nParticles)
{
  GRID_STRIDE(index, stride);
  for (int i = index; i < nParticles; i += stride)
  {
    vec3i cellIndex = getCellIndex(d_particles[i].pos, cellSize, nCellsPerSide);
    int cellId = getCellIndex1D(cellIndex, nCellsPerSide);
    PartDS[i] = atomicExch(&CellDs[cellId], i);
  }
}

// CUDA kernel to compute densities
__global__ void computeDensitiesKernel(Particle *d_particles,
                                       int *d_cellHead,
                                       int *d_cellNext,
                                       double cellSize,
                                       int nCellsPerSide,
                                       double h,
                                       int nParticles,
                                       double restDensity,
                                       double gasConstant)
{
  double weightA = 315.0 / (64.0 * M_PI * pow(h, 9));
  double h2 = h * h;

  GRID_STRIDE(index, stride);

  for (int i = index; i < nParticles; i += stride)
  {
    double density = restDensity;
    if (d_particles[i].fix == false)
    {
      density = 0;
      vec3i cellIndex = getCellIndex(d_particles[i].pos, cellSize, nCellsPerSide);

      for (int ix = -1; ix <= 1; ++ix)
      {
        for (int iy = -1; iy <= 1; ++iy)
        {
          for (int iz = -1; iz <= 1; ++iz)
          {
            int3 neighborCellIndex = make_int3((cellIndex.x + ix + nCellsPerSide) % nCellsPerSide,
                                               (cellIndex.y + iy + nCellsPerSide) % nCellsPerSide,
                                               (cellIndex.z + iz + nCellsPerSide) % nCellsPerSide);

            int neighborCellId = getCellIndex1D(neighborCellIndex, nCellsPerSide);

            int neighborIdx = d_cellHead[neighborCellId];
            while (neighborIdx != -1)
            {
              double r2 = normSquared(d_particles[i].pos - d_particles[neighborIdx].pos);
              double weightB = pow(h2 - r2, 3);

              if (r2 <= h2)
              {
                density += d_particles[neighborIdx].mass * weightA * weightB;
              }

              neighborIdx = d_cellNext[neighborIdx];
            }
          }
        }
      }
    }
    d_particles[i].density = density;
    d_particles[i].pressure = gasConstant * (density - restDensity); 
  }
}

// CUDA kernel to compute forces
__global__ void calculateForcesKernel(Particle *d_particles, int *d_cellHead, int *d_cellNext, double cellSize, int nCellsPerSide, double h, int nParticles, double viscosityCoefficient, double gravity)
{
  double weightA = -45.0 / (M_PI * pow(h, 6));

  GRID_STRIDE(index, stride);

  for (int i = index; i < nParticles; i += stride)
  {
    if (d_particles[i].fix == false)
    {
      double fx = 0.0;
      double fy = 0.0;
      double fz = 0.0;

      vec3i cellIndex = getCellIndex(d_particles[i].pos, cellSize, nCellsPerSide);

      for (int dx = -1; dx <= 1; ++dx)
      {
        for (int dy = -1; dy <= 1; ++dy)
        {
          for (int dz = -1; dz <= 1; ++dz)
          {
            vec3i neighborCellIndex = make_int3((cellIndex.x + dx + nCellsPerSide) % nCellsPerSide,
                                                (cellIndex.y + dy + nCellsPerSide) % nCellsPerSide,
                                                (cellIndex.z + dz + nCellsPerSide) % nCellsPerSide);

            int neighborCellId = getCellIndex1D(neighborCellIndex, nCellsPerSide);
            int neighborIdx = d_cellHead[neighborCellId];

            while (neighborIdx != -1)
            {
              if (neighborIdx != i)
              {
                double dist_x = d_particles[i].pos.x - d_particles[neighborIdx].pos.x;
                double dist_y = d_particles[i].pos.y - d_particles[neighborIdx].pos.y;
                double dist_z = d_particles[i].pos.z - d_particles[neighborIdx].pos.z;

                double r = sqrt(dist_x * dist_x + dist_y * dist_y + dist_z * dist_z);

                if (r <= h)
                {
                  // Pressure force
                  double pressureTerm = -0.5 * d_particles[neighborIdx].mass * (d_particles[i].pressure + d_particles[neighborIdx].pressure) / d_particles[neighborIdx].density;
                  double weightB = h - r;
                  double pressureGradient = weightA * weightB * weightB;
                  fx += pressureTerm * pressureGradient * dist_x / (r); // + 0.01 * h * h);
                  fy += pressureTerm * pressureGradient * dist_y / (r); // + 0.01 * h * h);
                  fz += pressureTerm * pressureGradient * dist_z / (r); // + 0.01 * h * h);

                  // Viscosity force
                  if (d_particles[neighborIdx].fix == false)
                  {
                    double viscosityLaplacian = -weightA * weightB;
                    double viscosityTerm_x = viscosityCoefficient * d_particles[neighborIdx].mass * (d_particles[neighborIdx].vel.x - d_particles[i].vel.x) / d_particles[neighborIdx].density;
                    double viscosityTerm_y = viscosityCoefficient * d_particles[neighborIdx].mass * (d_particles[neighborIdx].vel.y - d_particles[i].vel.y) / d_particles[neighborIdx].density;
                    double viscosityTerm_z = viscosityCoefficient * d_particles[neighborIdx].mass * (d_particles[neighborIdx].vel.z - d_particles[i].vel.z) / d_particles[neighborIdx].density;
                    fx += viscosityTerm_x * viscosityLaplacian;
                    fy += viscosityTerm_y * viscosityLaplacian;
                    fz += viscosityTerm_z * viscosityLaplacian;
                  }
                }
              }
              neighborIdx = d_cellNext[neighborIdx];
            }
          }
        }
      }

      d_particles[i].force.x = fx;
      d_particles[i].force.y = fy - gravity * d_particles[i].density;
      d_particles[i].force.z = fz;
    }
  }
}

// CUDA kernel for velocity integration
__global__ void updateState(Particle *d_particles, double dt, int n)
{
  GRID_STRIDE(index, stride);
  for (int i = index; i < n; i += stride)
  {
    if (d_particles[i].fix == false)
    {
      d_particles[i].vel += d_particles[i].force * (dt / d_particles[i].density);
      d_particles[i].pos += d_particles[i].vel * dt;
    }
  }
}


