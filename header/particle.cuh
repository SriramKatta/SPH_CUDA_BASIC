#pragma once

using vec3d = double3;

struct Particle
{
  vec3d pos = make_double3(0.0, 0.0, 0.0);   // Position
  vec3d vel = make_double3(0.0, 0.0, 0.0);   // Velocity
  vec3d force = make_double3(0.0, 0.0, 0.0); // Force
  double mass{};                               // mass
  double radius{};                             // radius
  bool fix{};                                  // boundary type
  double density{};                            // density
  double pressure{};                           // pressure
};
