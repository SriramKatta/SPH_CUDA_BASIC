#pragma once

#include "particle.cuh"
#include "thrust/host_vector.h"

thrust::host_vector<Particle> sceneInit(const double &boxSize, const double &h, const int &nDompart)
{
  thrust::host_vector<Particle> particles;

  // place particles on box
  const int resolution = boxSize / h * 2;
  particles.reserve(6 * resolution * resolution);
  for (int iz = 1; iz < resolution; ++iz){
    const double zpos = static_cast<double>(iz)/resolution * boxSize;
    for (int ix = 1; ix < resolution; ++ix)
    {
      const double xpos = static_cast<double>(ix)/resolution * boxSize;
      const double ypos1 = 0.0;
      const double ypos2 = boxSize;
      Particle p1, p2;
      p1.ghost = true;
      p2.ghost = true;
      p1.pos = make_double3(xpos, ypos1, zpos);
      p2.pos = make_double3(xpos, ypos2, zpos);
      particles.push_back(p1);
      particles.push_back(p2);
    }
  }
}