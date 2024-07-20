#pragma once

#include <cmath>

#include "cuda_helper.cuh"
#include "vector_helper.cuh"


using vec3d = double3;

class kernel{
public:
  HOST_DEVICE_FUNC
  kernel(double rad) : m_rad(rad){
    m_a = 315.0 / (64.0 * M_PI * pow(rad, 9));
    m_b = 45.0 / (M_PI * pow(rad, 6));
  }

  HOST_DEVICE_FUNC
  double W(const vec3d &r){
    return W(norm(r));
  }

  HOST_DEVICE_FUNC
  double W(const double& r){
    if (r > m_rad){
      return 0.0;
    }
    return m_a * pow(m_rad * m_rad - r * r, 3);
  }


private:
  double m_rad;
  double m_a;
  double m_b;
};