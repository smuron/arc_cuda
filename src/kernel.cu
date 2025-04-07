#include "kernel.h"
#include "simulation.h"
#include <cmath>
#include <cuda_runtime.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <utility>

#define CHECK_CUDA_ERROR(err, msg)                                             \
  do {                                                                         \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, msg);  \
      fprintf(stderr, "Error code: %d, Message: %s\n", err_,                   \
              cudaGetErrorString(err_));                                       \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

struct SimulationInstance::CUDAData {
  targetData *targetDevice;
  VoxelData *targetVoxelsDevice;
  particleData *particlesDevice;
  dim3 threadsPerBlock;
  dim3 blocksPerGrid;
};

extern "C" void setupSimulation(SimulationInstance *sim) {
  CHECK_CUDA_ERROR(cudaDeviceReset(), "device reset");
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  printf("CUDA Memory - Free: %zu B, Total: %zu B, Free MB: %zu MB, Total MB: "
         "%zu MB, Request: %zu B\n",
         free, total, free / (1024 * 1024), total / (1024 * 1024),
         sizeof(targetData));
  sim->cuda_data = new SimulationInstance::CUDAData();

  // Allocate and copy target structure
  cudaDeviceSynchronize();
  CHECK_CUDA_ERROR(cudaGetLastError(), "before targetDevice malloc");
  CHECK_CUDA_ERROR(
      cudaMalloc(&sim->cuda_data->targetDevice, sizeof(targetData)),
      "malloc targetDevice");

  cudaDeviceSynchronize();
  CHECK_CUDA_ERROR(cudaGetLastError(), "before voxelsDevice malloc");
  CHECK_CUDA_ERROR(
      cudaMalloc(&sim->cuda_data->targetVoxelsDevice,
                 sim->target->voxel_size * sim->target->voxel_size *
                     sim->target->voxel_size * sizeof(VoxelData)),
      "malloc voxelsDevice");

  targetData hostTargetCopy = targetData(sim->target);
  hostTargetCopy.voxels =
      sim->cuda_data->targetVoxelsDevice; // Point to device memory
  CHECK_CUDA_ERROR(cudaMemcpy(sim->cuda_data->targetDevice, &hostTargetCopy,
                              sizeof(targetData), cudaMemcpyHostToDevice),
                   "memcpy target");
  CHECK_CUDA_ERROR(
      cudaMemcpy(sim->cuda_data->targetVoxelsDevice, sim->target->voxels,
                 sim->target->voxel_size * sim->target->voxel_size *
                     sim->target->voxel_size * sizeof(VoxelData),
                 cudaMemcpyHostToDevice),
      "memcpy voxels");

  // Allocate and copy particles
  CHECK_CUDA_ERROR(cudaMalloc(&sim->cuda_data->particlesDevice,
                              sim->num_particles * sizeof(particleData)),
                   "malloc particleData");
  CHECK_CUDA_ERROR(cudaMemcpy(sim->cuda_data->particlesDevice, sim->particles,
                              sim->num_particles * sizeof(particleData),
                              cudaMemcpyHostToDevice),
                   "memcpy particleData");

  // Set up grid dimensions
  sim->cuda_data->threadsPerBlock = dim3(256, 1, 1);
  sim->cuda_data->blocksPerGrid =
      dim3((sim->num_particles + sim->cuda_data->threadsPerBlock.x - 1) /
               sim->cuda_data->threadsPerBlock.x,
           1, 1);
}

extern "C" void teardownSimulation(SimulationInstance *sim) {
  if (sim->cuda_data) {
    cudaFree(sim->cuda_data->targetVoxelsDevice);
    cudaFree(sim->cuda_data->targetDevice);
    cudaFree(sim->cuda_data->particlesDevice);
    delete sim->cuda_data;
    sim->cuda_data = nullptr;
  }
}

__device__ float3 arrayToFloat3(float arr[3]) {
  return make_float3(arr[0], arr[1], arr[2]);
}
__device__ float3 arrayToFloat3(const float arr[3]) {
  return make_float3(arr[0], arr[1], arr[2]);
}

__device__ float3 operator*(const float3 &v, float t) {
  return make_float3(v.x * t, v.y * t, v.z * t);
}

__device__ float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator/(const float3 &v, float t) {
  return make_float3(v.x / t, v.y / t, v.z / t);
}

__device__ float3 operator/(const float3 &a, const float3 &b) {
  return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

template <typename T> __device__ void swap(T &a, T &b) {
  T c(a);
  a = b;
  b = c;
}

struct TraversalResult {
  bool hit;
  float hit_time;
  int3 voxel_coord;
};

__device__ bool intersectGrid(const float3 &origin, const float3 &direction,
                              const float3 &box_min, const float3 &box_max,
                              float &t_entry, float &t_exit) {
  float t_near_x = (box_min.x - origin.x) / direction.x;
  float t_far_x = (box_max.x - origin.x) / direction.x;
  if (direction.x < 0)
    swap(t_near_x, t_far_x);
  if (fabsf(direction.x) < 1e-6f) {
    t_near_x = (origin.x < box_min.x) ? INFINITY : -INFINITY;
    t_far_x = (origin.x > box_max.x) ? -INFINITY : INFINITY;
  }

  float t_near_y = (box_min.y - origin.y) / direction.y;
  float t_far_y = (box_max.y - origin.y) / direction.y;
  if (direction.y < 0)
    swap(t_near_y, t_far_y);
  if (fabsf(direction.y) < 1e-6f) {
    t_near_y = (origin.y < box_min.y) ? INFINITY : -INFINITY;
    t_far_y = (origin.y > box_max.y) ? -INFINITY : INFINITY;
  }

  float t_near_z = (box_min.z - origin.z) / direction.z;
  float t_far_z = (box_max.z - origin.z) / direction.z;
  if (direction.z < 0)
    swap(t_near_z, t_far_z);
  if (fabsf(direction.z) < 1e-6f) {
    t_near_z = (origin.z < box_min.z) ? INFINITY : -INFINITY;
    t_far_z = (origin.z > box_max.z) ? -INFINITY : INFINITY;
  }

  // printf(
  //     "intersectGrid: t_x=(%f,%f), t_y=(%f,%f), t_z=(%f,%f),
  //     dir=(%f,%f,%f)\n", t_near_x, t_far_x, t_near_y, t_far_y, t_near_z,
  //     t_far_z, direction.x, direction.y, direction.z);

  t_entry = fmaxf(t_near_x, fmaxf(t_near_y, t_near_z));
  t_exit = fminf(t_far_x, fminf(t_far_y, t_far_z));

  if (t_entry > t_exit || t_exit < 0) {
    return false;
  }
  return true;
}

__device__ inline int voxelIndex(int3 coord, int3 grid_size) {
  return coord.z * (grid_size.x * grid_size.y) + coord.y * grid_size.x +
         coord.x;
}

__device__ TraversalResult
voxel_traverse(const VoxelData *voxels, const float3 &origin,
               const float3 &direction, const float3 &box_min,
               const float3 &box_max, const int3 grid_size, float max_time) {
  TraversalResult result = {false, 0.0f, make_int3(0, 0, 0)};
  float t_entry, t_exit;

  if (!intersectGrid(origin, direction, box_min, box_max, t_entry, t_exit)) {
    return result;
  }

  t_exit = fminf(t_exit, max_time);
  if (t_entry > t_exit) {
    // printf("entry > exit %f %f %f", t_entry, t_exit, max_time);
    return result;
  }
  float current_t = 0.0f;
  if (t_entry > 0.0f) {
    current_t = t_entry;
  }
  float3 pos = origin + direction * t_entry;

  float3 grid_size_f = make_float3(grid_size.x, grid_size.y, grid_size.z);
  float3 cell_size = (box_max - box_min) / grid_size_f;

  int3 voxel = make_int3((pos.x - box_min.x) / cell_size.x,
                         (pos.y - box_min.y) / cell_size.y,
                         (pos.z - box_min.z) / cell_size.z);

  // printf("Traversal start: pos=(%f,%f,%f), voxel=(%d,%d,%d), t_entry=%f\n",
  //        pos.x, pos.y, pos.z, voxel.x, voxel.y, voxel.z, t_entry);

  int3 step = make_int3(direction.x > 0 ? 1 : -1, direction.y > 0 ? 1 : -1,
                        direction.z > 0 ? 1 : -1);
  float3 t_delta = make_float3(fabsf(cell_size.x / direction.x),
                               fabsf(cell_size.y / direction.y),
                               fabsf(cell_size.z / direction.z));
  float3 t_next =
      make_float3(((step.x > 0 ? voxel.x + 1 : voxel.x) * cell_size.x +
                   box_min.x - origin.x) /
                      direction.x,
                  ((step.y > 0 ? voxel.y + 1 : voxel.y) * cell_size.y +
                   box_min.y - origin.y) /
                      direction.y,
                  ((step.z > 0 ? voxel.z + 1 : voxel.z) * cell_size.z +
                   box_min.z - origin.z) /
                      direction.z);

  int step_count = 0;
  while (current_t <= t_exit) {
    if (voxel.x >= 0 && voxel.x < grid_size.x && voxel.y >= 0 &&
        voxel.y < grid_size.y && voxel.z >= 0 && voxel.z < grid_size.z) {
      int idx = voxelIndex(voxel, grid_size);
      if (voxels[idx].density > 0.0) {
        // printf("Hit at voxel=(%d,%d,%d), pos=(%f,%f,%f), t=%f, steps=%d\n",
        //        voxel.x, voxel.y, voxel.z, pos.x, pos.y, pos.z, current_t,
        //        step_count);
        result.hit = true;
        result.hit_time = current_t;
        result.voxel_coord = voxel;
        return result;
      }
    }
    step_count++;
    if (t_next.x < t_next.y && t_next.x < t_next.z) {
      voxel.x += step.x;
      current_t = t_next.x;
      t_next.x += t_delta.x;
    } else if (t_next.y < t_next.z) {
      voxel.y += step.y;
      current_t = t_next.y;
      t_next.y += t_delta.y;
    } else {
      voxel.z += step.z;
      current_t = t_next.z;
      t_next.z += t_delta.z;
    }
    pos = origin + direction * current_t; // Update pos for debug
  }
  // printf("No hit, final voxel=(%d,%d,%d), pos=(%f,%f,%f), steps=%d\n",
  //        voxel.x, voxel.y, voxel.z, pos.x, pos.y, pos.z, step_count);
  return result;
}
const float PARTICLE_SPECIFIC_HEAT = 2.0f; // or whatever
const float PARTICLE_MASS = 1e-4f;         // very small mass
const float TARGET_MELTING_POINT = 1500.0f;
const float TARGET_HEAT_CAPACITY = 0.4f;
const float TARGET_HEAT_DISSIPATION = 0.222f;

__device__ float calculateParticleTemperature(SimParameters params, float lifetime) {
  return params.particle_start_temperature * (1.0f - lifetime) +
         params.particle_end_temperature * lifetime;
}

__device__ float calculateParticleEnergy(SimParameters params, float mass, float3 velocity,
                                         float temperature) {
  float kinetic = 0.5f * mass *
                  (velocity.x * velocity.x + velocity.y * velocity.y +
                   velocity.z * velocity.z);
  float thermal = mass * PARTICLE_SPECIFIC_HEAT * temperature;
  return kinetic + thermal;
}

__device__ void applyDamage(VoxelData *voxels, int idx, float damage,
                            float particle_energy) {
  // First add heat
  float old_temp = atomicAdd(&voxels[idx].temperature,
                             particle_energy / TARGET_HEAT_CAPACITY);

  // printf("applyDamage voxels %ul %f %f %f", idx, damage, particle_energy,
  //        old_temp);
  //  Only apply damage if temperature is at/above melting point
  if (old_temp >= TARGET_MELTING_POINT) {
    // printf("target is melting at voxel %ul %f %f\n", idx,
    // voxels[idx].density, damage);
    float old_density = atomicAdd(&voxels[idx].density, -damage);

    // Prevent negative density
    if (old_density <= damage) {
      // printf("density becoming zero at %ul\n", idx);
      voxels[idx].dirty = true;
    }
  }
}

__global__ void particle_update(particleData *particles, int particle_count,
                                targetData *target, float delta_time, SimParameters params) {
  int grid_size = target->voxel_size;
  int3 grid_size3 = make_int3(grid_size, grid_size, grid_size);
  VoxelData *voxels = target->voxels;
  float3 box_min =
      make_float3(-target->world_size / 2.0f, -target->world_size / 2.0f,
                  -target->world_size / 2.0f);
  float3 box_max =
      make_float3(target->world_size / 2.0f, target->world_size / 2.0f,
                  target->world_size / 2.0f);
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= particle_count || !particles[i].is_active)
    return;

  float3 pos = arrayToFloat3(particles[i].position);
  float3 vel = arrayToFloat3(particles[i].velocity);

  TraversalResult result = voxel_traverse(voxels, pos, vel, box_min, box_max,
                                          grid_size3, delta_time);
  if (result.hit) {
    int voxel_idx = voxelIndex(result.voxel_coord, grid_size3);
    target->voxels[voxel_idx].dirty = true;

    // printf("hit %ul", voxel_idx);
    float particle_energy = calculateParticleEnergy(params,
        PARTICLE_MASS, vel,
        calculateParticleTemperature(params, particles[i].lifetime));
    const float DAMAGE_FACTOR = 0.01f;
    float damage = particle_energy * DAMAGE_FACTOR;
    applyDamage(voxels, voxel_idx, damage, particle_energy);
    particles[i].velocity[0] = 0.0f;
    particles[i].velocity[1] = 0.0f;
    particles[i].velocity[2] = 0.0f;

    particles[i].is_active = false;
    return;
  }

  float3 acc = arrayToFloat3(particles[i].acceleration);
  vel = vel + acc * delta_time;
  pos = pos + vel * delta_time;

  particles[i].lifetime += delta_time;
  if (particles[i].lifetime >= 2.0f) {
    particles[i].is_active = false;
    particles[i].velocity[0] = 0.0f;
    particles[i].velocity[1] = 0.0f;
    particles[i].velocity[2] = 0.0f;
    return;
  }
  particles[i].position[0] = pos.x;
  particles[i].position[1] = pos.y;
  particles[i].position[2] = pos.z;
  particles[i].velocity[0] = vel.x;
  particles[i].velocity[1] = vel.y;
  particles[i].velocity[2] = vel.z;
}

__global__ void voxel_update(targetData *target, float delta_time) {
  // Get 3D thread indices
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= target->voxel_size || y >= target->voxel_size ||
      z >= target->voxel_size)
    return;

  int idx =
      z * target->voxel_size * target->voxel_size + y * target->voxel_size + x;
  target->voxels[idx].dirty = false;
  if (target->voxels[idx].density <= 0.0) {
    target->voxels[idx].density = 0.0f;
    return;
  }
  // Simple heat dissipation to neighbors
  float heat_delta = 0.0f;
  const float dissipation_rate = TARGET_HEAT_DISSIPATION * delta_time;

  // Check 6 neighbors
  const int neighbors[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                               {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};

  for (int n = 0; n < 6; n++) {
    int nx = x + neighbors[n][0];
    int ny = y + neighbors[n][1];
    int nz = z + neighbors[n][2];

    if (nx >= 0 && nx < target->voxel_size && ny >= 0 &&
        ny < target->voxel_size && nz >= 0 && nz < target->voxel_size) {

      int nidx = nz * target->voxel_size * target->voxel_size +
                 ny * target->voxel_size + nx;

      float temp_diff =
          target->voxels[nidx].temperature - target->voxels[idx].temperature;
      heat_delta += temp_diff * dissipation_rate;
    }
  }

  if (heat_delta != 0.0) {
    target->voxels[idx].dirty = true;
    atomicAdd(&target->voxels[idx].temperature, heat_delta);
  }
}

__global__ void spawn_particles(particleData *particles, int num_particles,
                                float *nozzle_pos, float *nozzle_vel,
                                float *nozzle_acc, float velocity_spread,
                                float acceleration_spread, float angle,
                                float delta_time, unsigned int seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_particles)
    return;

  if (particles[idx].is_active)
    return;

  // Simple random number generation
  unsigned int rand_state = seed + idx;
  rand_state = rand_state * 1664525 + 1013904223;
  float rand1 = rand_state / (float)UINT_MAX;
  rand_state = rand_state * 1664525 + 1013904223;
  float rand2 = rand_state / (float)UINT_MAX;

  // Random angle within cone
  float theta = rand1 * 2 * PI;
  float phi = rand2 * angle;

  // Calculate direction vector
  float sin_phi = sinf(phi);
  float dir_x = sin_phi * cosf(theta);
  float dir_y = sin_phi * sinf(theta);
  float dir_z = cosf(phi);

  // Set particle properties
  particles[idx].is_active = true;
  particles[idx].lifetime = 0.0f;

  // Position
  particles[idx].position[0] = nozzle_pos[0];
  particles[idx].position[1] = nozzle_pos[1];
  particles[idx].position[2] = nozzle_pos[2];

  // Velocity with spread
  particles[idx].velocity[0] = nozzle_vel[0] + dir_x * velocity_spread;
  particles[idx].velocity[1] = nozzle_vel[1] + dir_y * velocity_spread;
  particles[idx].velocity[2] = nozzle_vel[2] + dir_z * velocity_spread;

  // Acceleration with spread
  particles[idx].acceleration[0] = nozzle_acc[0] + dir_x * acceleration_spread;
  particles[idx].acceleration[1] = nozzle_acc[1] + dir_y * acceleration_spread;
  particles[idx].acceleration[2] = nozzle_acc[2] + dir_z * acceleration_spread;
}

extern "C" void computeAndRenderSimulation(SimulationInstance *sim,
                                           float delta_time) {
  dim3 threadsPerBlockVoxel(8, 8, 8); // 8x8x8 = 512 threads per block
  dim3 blocksPerGridVoxel(
      (sim->target->voxel_size + threadsPerBlockVoxel.x - 1) /
          threadsPerBlockVoxel.x,
      (sim->target->voxel_size + threadsPerBlockVoxel.y - 1) /
          threadsPerBlockVoxel.y,
      (sim->target->voxel_size + threadsPerBlockVoxel.z - 1) /
          threadsPerBlockVoxel.z);
  voxel_update<<<blocksPerGridVoxel, threadsPerBlockVoxel>>>(
      sim->cuda_data->targetDevice, delta_time);
  CHECK_CUDA_ERROR(cudaGetLastError(), "after voxel_update kernel launch");
  CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "sync after voxel_update");

  particle_update<<<sim->cuda_data->blocksPerGrid,
                    sim->cuda_data->threadsPerBlock>>>(
      sim->cuda_data->particlesDevice, sim->num_particles,
      sim->cuda_data->targetDevice, delta_time, sim->params);
  CHECK_CUDA_ERROR(cudaGetLastError(), "after kernel last error check");
  CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "device sync");

  // Update parameter access
  float host_nozzle_pos[3];
  float host_nozzle_vel[3];
  float host_nozzle_acc[3];
  memcpy(host_nozzle_pos, sim->params.nozzle_origin, sizeof(float) * 3);
  memcpy(host_nozzle_vel, sim->params.nozzle_initial_velocity,
         sizeof(float) * 3);
  memcpy(host_nozzle_acc, sim->params.nozzle_initial_acceleration,
         sizeof(float) * 3);

  float velocity_spread = sim->params.nozzle_velocity_spread;
  float acceleration_spread = sim->params.nozzle_acceleration_spread;
  float angle = sim->params.nozzle_angle;
  unsigned int seed = static_cast<unsigned int>(time(nullptr));

  // Allocate device memory for nozzle parameters
  float *d_nozzle_pos, *d_nozzle_vel, *d_nozzle_acc;
  CHECK_CUDA_ERROR(cudaMalloc(&d_nozzle_pos, 3 * sizeof(float)),
                   "malloc nozzle_pos");
  CHECK_CUDA_ERROR(cudaMalloc(&d_nozzle_vel, 3 * sizeof(float)),
                   "malloc nozzle_vel");
  CHECK_CUDA_ERROR(cudaMalloc(&d_nozzle_acc, 3 * sizeof(float)),
                   "malloc nozzle_acc");

  // Copy host data to device
  CHECK_CUDA_ERROR(cudaMemcpy(d_nozzle_pos, host_nozzle_pos, 3 * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "memcpy nozzle_pos");
  CHECK_CUDA_ERROR(cudaMemcpy(d_nozzle_vel, host_nozzle_vel, 3 * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "memcpy nozzle_vel");
  CHECK_CUDA_ERROR(cudaMemcpy(d_nozzle_acc, host_nozzle_acc, 3 * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "memcpy nozzle_acc");

  // Launch kernel
  spawn_particles<<<sim->cuda_data->blocksPerGrid,
                    sim->cuda_data->threadsPerBlock>>>(
      sim->cuda_data->particlesDevice, sim->num_particles, d_nozzle_pos,
      d_nozzle_vel, d_nozzle_acc, velocity_spread, acceleration_spread, angle,
      delta_time, seed);
  CHECK_CUDA_ERROR(cudaGetLastError(), "after spawn_particles kernel launch");
  CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "sync after spawn_particles");

  // Free device memory
  CHECK_CUDA_ERROR(cudaFree(d_nozzle_pos), "free nozzle_pos");
  CHECK_CUDA_ERROR(cudaFree(d_nozzle_vel), "free nozzle_vel");
  CHECK_CUDA_ERROR(cudaFree(d_nozzle_acc), "free nozzle_acc");
}

extern "C" void cudaTargetToHost(targetData *host_target,
                                 SimulationInstance *sim) {
  CHECK_CUDA_ERROR(
      cudaMemcpy(host_target->voxels, sim->cuda_data->targetVoxelsDevice,
                 host_target->voxel_size * host_target->voxel_size *
                     host_target->voxel_size * sizeof(VoxelData),
                 cudaMemcpyDeviceToHost),
      "memcpy voxels D2H");
}

extern "C" void cudaParticlesToHost(particleData *host_particles,
                                    SimulationInstance *sim) {
  CHECK_CUDA_ERROR(cudaMemcpy(host_particles, sim->cuda_data->particlesDevice,
                              sim->num_particles * sizeof(particleData),
                              cudaMemcpyDeviceToHost),
                   "memcpy particles D2H");
}


extern "C" void resetCudaData(SimulationInstance *sim) {
   SimulationInstance::CUDAData* cuda_data = sim->cuda_data;
   targetData* target = sim->target;
   particleData* particles = sim->particles; 
   if (cuda_data) {
        // Copy reset particle data to device
        CHECK_CUDA_ERROR(cudaMemcpy(cuda_data->particlesDevice, particles,
                                   sim->num_particles * sizeof(particleData),
                                   cudaMemcpyHostToDevice),
                        "memcpy particles H2D during reset");

        // Copy reset target data to device
        targetData hostTargetCopy = targetData(target);
        hostTargetCopy.voxels = cuda_data->targetVoxelsDevice;  // Point to device memory
        CHECK_CUDA_ERROR(cudaMemcpy(cuda_data->targetDevice, &hostTargetCopy,
                                   sizeof(targetData), cudaMemcpyHostToDevice),
                        "memcpy target H2D during reset");
        CHECK_CUDA_ERROR(cudaMemcpy(cuda_data->targetVoxelsDevice, target->voxels,
                                   target->voxel_size * target->voxel_size * target->voxel_size * sizeof(VoxelData),
                                   cudaMemcpyHostToDevice),
                        "memcpy voxels H2D during reset");
    }
}
