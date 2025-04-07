#include "simulation.h"
#include "kernel.h"
#include "server.h"
#include <cstddef>
#include <cstdio>
#include <math.h>
#include <memory.h>
#include <mutex>
#include <stdio.h>
#include <thread>

// Define the static constants
const float targetData::MELTING_POINT = 1500.0f;
const float targetData::HEAT_CAPACITY = 0.5f;
const float targetData::HEAT_DISSIPATION = 0.1f;

targetData::targetData(float start_temp) {
  origin[0] = origin[1] = origin[2] = 0.0f;
  voxel_size = DEFAULT_TARGET_SIZE_VOXEL;
  world_size = DEFAULT_TARGET_SIZE_WORLD;

  const size_t total_voxels = voxel_size * voxel_size * voxel_size;
  voxels = new VoxelData[total_voxels];

  for (size_t i = 0; i < total_voxels; i++) {
    voxels[i] = VoxelData();
    voxels[i].density = 1.0f;
    voxels[i].temperature = start_temp;
  }
}

targetData::targetData() {
  origin[0] = origin[1] = origin[2] = 0.0f;
  voxel_size = DEFAULT_TARGET_SIZE_VOXEL;
  world_size = DEFAULT_TARGET_SIZE_WORLD;

  const size_t total_voxels = voxel_size * voxel_size * voxel_size;
  voxels = new VoxelData[total_voxels];

  // Create a test pattern
  for (size_t z = 0; z < voxel_size; z++) {
    for (size_t y = 0; y < voxel_size; y++) {
      for (size_t x = 0; x < voxel_size; x++) {
        size_t idx = z * voxel_size * voxel_size + y * voxel_size + x;

        // Create a sphere in the middle
        float cx = static_cast<float>(x) - voxel_size / 2.0f;
        float cy = static_cast<float>(y) - voxel_size / 2.0f;
        float cz = static_cast<float>(z) - voxel_size / 2.0f;
        float dist = sqrt(cx * cx + cy * cy + cz * cz);
        if (dist <= 2.0f || idx == voxel_size * voxel_size * voxel_size / 2) {
          if (idx == voxel_size * voxel_size * voxel_size / 2) {
            printf("\n\n");
          }
          printf("voxel init %f %f %f %f %f\n", cx, cy, cz, dist,
                 voxel_size / 4.0f);
        }

        if (dist < voxel_size / 2.0f || true) {
          voxels[idx].density = 1.0f;
          voxels[idx].temperature = 500.0f;
        } else {
          voxels[idx].density = 0.0f;
          voxels[idx].temperature = 300.0f;
        }
      }
    }
  }
  size_t center_idx = (voxel_size / 2) * voxel_size * voxel_size +
                      (voxel_size / 2) * voxel_size + (voxel_size / 2);
  printf("Direct center check: idx=%zu density=%f temp=%f\n", center_idx,
         voxels[center_idx].density, voxels[center_idx].temperature);
}

targetData::~targetData() {
  if (owns_voxels) {
    delete[] voxels;
  }
}

// Add copy constructor
targetData::targetData(const targetData *other) {
  memcpy(this, other, sizeof(targetData));
  owns_voxels = false; // Copy doesn't own the voxels
}

particleData::particleData() : is_active(false), lifetime(1.0f) {
  for (int i = 0; i < 3; i++) {
    position[i] = 0.0f;
    velocity[i] = 0.0f;
    acceleration[i] = 0.0f;
  }
}

SimulationInstance::SimulationInstance(unsigned int particle_count) 
    : num_particles(particle_count)
    , params() // Will use default values from struct
{
    target = new targetData();
    particles = new particleData[num_particles];

    // Initialize network buffers
    for (int i = 0; i < 2; i++) {
        network_buffers[i].particles = new particleData[num_particles];
        network_buffers[i].target = new targetData();
        network_buffers[i].target->voxels = 
            new VoxelData[target->voxel_size * target->voxel_size * target->voxel_size];
        network_buffers[i].ready = false;
    }

  // CUDA initialization can be handled in setupSimulation()
  cuda_data = nullptr;
}

SimulationInstance::~SimulationInstance() {
  delete target;
  delete[] particles;
  // CUDA cleanup handled in teardownSimulation()
  for (int i = 0; i < 2; i++) {
    delete[] network_buffers[i].particles;
    delete[] network_buffers[i].target->voxels;
    delete network_buffers[i].target;
  }
  if (network_running)
    stopNetworking();
}

void SimulationInstance::startNetworking() {
  network_running = true;
  network_thread = std::thread(&SimulationInstance::networkThreadFunc, this);
}
void SimulationInstance::stopNetworking() {
  network_running = false;
  if (network_thread.joinable())
    network_thread.join();
}

void SimulationInstance::networkThreadFunc() {
  while (network_running) {
    int buffer_idx = current_network_buffer;
    current_network_buffer = (current_network_buffer + 1) % 2;
    auto &buffer = network_buffers[buffer_idx];
    {
      std::lock_guard<std::mutex> lock(buffer.mutex);
      cudaParticlesToHost(buffer.particles, this);
      cudaTargetToHost(buffer.target, this);
      buffer.ready = true;
    }
    visServer->broadcastParticleData(buffer.particles, num_particles);
    visServer->broadcastVoxelData(buffer.target);
  }
  std::this_thread::sleep_for(
      std::chrono::milliseconds(static_cast<int>(network_update_rate * 1000)));
}

void SimulationInstance::updateParameters(const SimParameters& new_params) {
   params = new_params; 
}
