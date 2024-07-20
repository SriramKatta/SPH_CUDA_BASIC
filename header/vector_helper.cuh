#pragma once

#include "cuda_helper.cuh"
#include "cuda_runtime.h"

using vec3d = double3;
using vec3i = int3;

HOST_DEVICE_FUNC
vec3d operator+(const vec3d& a, const vec3d &b)
{
  return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

HOST_DEVICE_FUNC
int3 operator+(const int3 &a, const int3 &b)
{
  return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}


HOST_DEVICE_FUNC
vec3d operator-(const vec3d &a, const vec3d &b)
{
  return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

HOST_DEVICE_FUNC
int3 operator-(const int3 &a, const int3 &b)
{
  return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}

HOST_DEVICE_FUNC
vec3d operator/(const vec3d &a, const vec3d &b)
{
  return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
}

HOST_DEVICE_FUNC
vec3d operator/(const vec3d &a, const double &b)
{
  return make_double3(a.x / b, a.y / b, a.z / b);
}


HOST_DEVICE_FUNC
vec3i operator/(const vec3i &a, const int &b)
{
  return make_int3(a.x / b, a.y / b, a.z / b);
}




HOST_DEVICE_FUNC
vec3i operator/(const vec3i &a, const vec3i &b)
{
  return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}



HOST_DEVICE_FUNC
vec3d operator*(const vec3d &a, const vec3d &b)
{
  return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}

HOST_DEVICE_FUNC
vec3d operator*(const vec3d &a, const double &b)
{
  return make_double3(a.x * b, a.y * b, a.z * b);
}

HOST_DEVICE_FUNC
int3 operator*(const int3 &a, const int3 &b)
{
  return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}

HOST_DEVICE_FUNC
int prod(const int3 &a)
{
  return a.x * a.y * a.z;
}

HOST_DEVICE_FUNC
int3 casttoint(const vec3d &a)
{
  return make_int3(a.x, a.y, a.z);
}

HOST_DEVICE_FUNC
double3 casttodouble(const int3 &a)
{
  return make_double3(a.x, a.y, a.z);
}

HOST_DEVICE_FUNC
int3 ceil(const vec3d &a)
{
  return make_int3(ceil(a.x), ceil(a.y), ceil(a.z));
}

HOST_DEVICE_FUNC
double normSquared(const vec3d &a)
{
  return a.x * a.x + a.y * a.y + a.z * a.z;
}

HOST_DEVICE_FUNC
double norm(const vec3d &a)
{
  return sqrt(normSquared(a));
}

HOST_DEVICE_FUNC
vec3d normalised(const vec3d &a)
{
  double n = norm(a);
  return make_double3(a.x / n, a.y / n, a.z / n);
}

HOST_DEVICE_FUNC
int3 vecmod(const int3 &a, const int3 &b)
{
  return make_int3(a.x % b.x, a.y % b.y, a.z % b.z);
}

HOST_DEVICE_FUNC
vec3d operator+=(vec3d &a, const vec3d &b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}