#pragma once

#include "particle.cuh"
#include "vector_helper.cuh"
#include "kernel.cuh"

#include <cfloat>

using vec3d = double3;
using vec3i = int3;

__global__ void setMass(Particle *d_particles, double mass, int N)
{
  GRID_STRIDE(index, stride);
  for (int i = index; i < N; i += stride)
  {
    d_particles[i].mass = mass;
  }
}

__device__ vec3i getCellIndex(const vec3d &pos, const double &cellSize, const int &nXYZCells)
{
  vec3i cellIndex = make_int3(
      static_cast<int>(pos.x / cellSize + nXYZCells) % nXYZCells,
      static_cast<int>(pos.y / cellSize + nXYZCells) % nXYZCells,
      static_cast<int>(pos.z / cellSize + nXYZCells) % nXYZCells);
  return cellIndex;
}

__device__ int getCellIndex1D(const vec3i &cellIndex, int nXYZCells)
{
  return cellIndex.x * nXYZCells * nXYZCells + cellIndex.y * nXYZCells + cellIndex.z;
}

__global__ void setDS(int *DS, int N)
{
  GRID_STRIDE(index, stride);
  for (int i = index; i < N; i += stride)
  {
    DS[i] = -1;
  }
}


__global__ void updateDS(Particle *d_particles,
                         int *CellDs,
                         int *PartDS,
                         double cellSize,
                         int nXYZCells,
                         int Npart)
{
  GRID_STRIDE(start, stride);
  for (int i = start; i < Npart; i += stride)
  {
    vec3i cellIndex = getCellIndex(d_particles[i].pos, cellSize, nXYZCells);
    int cellId = getCellIndex1D(cellIndex, nXYZCells);
    PartDS[i] = atomicExch(&CellDs[cellId], i);
  }
}

__global__ void updatePressureDensity(Particle *d_particles,
                                      int *CellDs,
                                      int *partDs,
                                      double cellSize,
                                      int nXYZCells,
                                      int Npart,
                                      double h,
                                      double restDensity,
                                      double gasConstant,
                                      kernel *k)
{

  GRID_STRIDE(start, stride);
  for (int i = start; i < Npart; i += stride)
  {
    if (d_particles[i].ghost)
    {
      d_particles[i].density = restDensity;
      d_particles[i].pressure = 0.0;
      continue;
    }

    double density = 0;
    vec3i cellIndex = getCellIndex(d_particles[i].pos, cellSize, nXYZCells);

    NBD_LOOP(
        vec3i neighborCellIndex = make_int3((cellIndex.x + ix + nXYZCells) % nXYZCells,
                                           (cellIndex.y + iy + nXYZCells) % nXYZCells,
                                           (cellIndex.z + iz + nXYZCells) % nXYZCells);

        int neighborCellId = getCellIndex1D(neighborCellIndex, nXYZCells);

        int neighborIdx = CellDs[neighborCellId];
        while (neighborIdx != -1) {
          const vec3d rij = d_particles[i].pos - d_particles[neighborIdx].pos;
          double r = norm(rij);
          if (r <= h)
          {
            density += d_particles[neighborIdx].mass * k->W(r);
          }
          neighborIdx = partDs[neighborIdx];
        })
    d_particles[i].density = density;
    d_particles[i].pressure = gasConstant * (density - restDensity);
  }
}

__device__ vec3d pressureForce(Particle *d_particles, int i, int neighborIdx, double h, kernel *k)
{
  vec3d force = make_double3(0.0, 0.0, 0.0);
  vec3d rij = d_particles[i].pos - d_particles[neighborIdx].pos;
  double r = norm(rij);
  if (r <= h && r > 0.0)
  {
    double pressureTerm = -0.5 * d_particles[neighborIdx].mass * (d_particles[i].pressure + d_particles[neighborIdx].pressure) / d_particles[neighborIdx].density;
    double pressureGradient = k->gradW(r);
    force = rij * (pressureTerm * pressureGradient / (r + DBL_EPSILON));
  }
  return force;
}

__device__ vec3d viscForce(Particle *d_particles, int i, int neighborIdx, double visccoeff, double h, kernel *k)
{
  vec3d force = make_double3(0.0, 0.0, 0.0);
  vec3d rij = d_particles[i].pos - d_particles[neighborIdx].pos;
  double r = norm(rij);
  if (r <= h)
  {
    double viscosityLaplacian = k->lapW(r);
    force = (d_particles[neighborIdx].vel - d_particles[i].vel) * (visccoeff * viscosityLaplacian * d_particles[neighborIdx].mass / d_particles[neighborIdx].density);
  }
  return force;
}

__global__ void updateAcceleration(Particle *d_particles,
                                   int *d_cellHead,
                                   int *d_cellNext,
                                   double cellSize,
                                   int nXYZCells,
                                   double h,
                                   int Npart,
                                   double visccoeff,
                                   double gravity,
                                   kernel *k)
{

  GRID_STRIDE(start, stride);

  for (int i = start; i < Npart; i += stride)
  {
    if (!d_particles[i].ghost)
    {
      vec3d force = make_double3(0.0, 0.0, 0.0);

      vec3i cellIndex = getCellIndex(d_particles[i].pos, cellSize, nXYZCells);

      NBD_LOOP(
          vec3i neighborCellIndex = make_int3((cellIndex.x + ix + nXYZCells) % nXYZCells,
                                              (cellIndex.y + iy + nXYZCells) % nXYZCells,
                                              (cellIndex.z + iz + nXYZCells) % nXYZCells);

          int neighborCellId = getCellIndex1D(neighborCellIndex, nXYZCells);
          int neighborIdx = d_cellHead[neighborCellId];

          while (neighborIdx != -1) {
            if (neighborIdx != i)
            {
              vec3d rij = d_particles[i].pos - d_particles[neighborIdx].pos;
              double r = norm(rij);

              if (r <= h)
              {
                force += pressureForce(d_particles, i, neighborIdx, h, k);
                // Viscosity force
                if (!d_particles[neighborIdx].ghost)
                {
                  force += viscForce(d_particles, i, neighborIdx, visccoeff, h, k);
                }
              }
            }
            neighborIdx = d_cellNext[neighborIdx];
          })
      d_particles[i].acc = force / d_particles[i].density;
      d_particles[i].acc.y -= gravity;
    }
  }
}

__global__ void updateState(Particle *d_particles, double dt, int Npart)
{
  GRID_STRIDE(start, stride);
  for (int i = start; i < Npart; i += stride)
  {
    if (!d_particles[i].ghost)
    {
      d_particles[i].vel += d_particles[i].acc * dt;
      d_particles[i].pos += d_particles[i].vel * dt;
    }
  }
}
