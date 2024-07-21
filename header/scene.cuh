#pragma once

#include "particle.cuh"
#include "thrust/host_vector.h"

thrust::host_vector<Particle> sceneInit(const double &boxSize, const double& domstart, const double& domend,const double &h,int &nDompart)
{
  thrust::host_vector<Particle> particles;

  // place particles on box
  const int resolutionbdy = boxSize / h * 3;
  particles.reserve(6 * (resolutionbdy - 1) * (resolutionbdy -1));

  for (int iz = 1; iz < resolutionbdy; ++iz){
    const double zpos = static_cast<double>(iz)/resolutionbdy * boxSize;
    for (int ix = 1; ix < resolutionbdy; ++ix)
    {
      const double xpos = static_cast<double>(ix)/resolutionbdy * boxSize;
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

  for(int ix = 1; ix < resolutionbdy; ++ix){
    const double xpos = static_cast<double>(ix)/resolutionbdy * boxSize;
    for(int iy = 1; iy < resolutionbdy; ++iy){
      const double ypos = static_cast<double>(iy)/resolutionbdy * boxSize;
      const double zpos1 = 0.0;
      const double zpos2 = boxSize;
      Particle p1, p2;
      p1.ghost = true;
      p2.ghost = true;
      p1.pos = make_double3(xpos, ypos, zpos1);
      p2.pos = make_double3(xpos, ypos, zpos2);
      particles.push_back(p1);
      particles.push_back(p2);
    }
  }

  for(int iy = 1; iy < resolutionbdy; ++iy){
    const double ypos = static_cast<double>(iy)/resolutionbdy * boxSize;
    for(int iz = 1; iz < resolutionbdy; ++iz){
      const double zpos = static_cast<double>(iz)/resolutionbdy * boxSize;
      const double xpos1 = 0.0;
      const double xpos2 = boxSize;
      Particle p1, p2;
      p1.ghost = true;
      p2.ghost = true;
      p1.pos = make_double3(xpos1, ypos, zpos);
      p2.pos = make_double3(xpos2, ypos, zpos);
      particles.push_back(p1);
      particles.push_back(p2);
    }
  }

  int numghostparticles = particles.size();

  double domsize = domend - domstart;
  int resdom = domsize / h * 4;

  for(int ix = 0; ix < resdom; ++ix)
  for(int iy = 0; iy < resdom; ++iy)
  for(int iz = 0; iz < resdom; ++iz){
    Particle p;
    p.ghost = false;
    const double xpos = domstart + domsize * static_cast<double>(ix)/resdom;
    const double ypos = domstart + domsize * static_cast<double>(iy)/resdom;
    const double zpos = domstart + domsize * static_cast<double>(iz)/resdom;
    p.pos = make_double3(xpos, ypos, zpos);
    particles.push_back(p);
  }

  nDompart = particles.size() - numghostparticles;
  return particles;
}