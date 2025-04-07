#ifndef KERNEL_H
#define KERNEL_H

#include <cmath>
#include <cstdint>
#include <memory.h>
#include <vector>


const float PI = 3.14159265f;
const float DEFAULT_TARGET_SIZE_WORLD = 1.0f;
const unsigned int DEFAULT_TARGET_SIZE_VOXEL = 64;

struct SimParameters {
  float nozzle_origin[3] = {0.0, 0.0, -2.5f};
  float nozzle_initial_acceleration[3] = {0.0, 0.0, 0.0};
  float nozzle_initial_velocity[3] = {0.0, 0.0, 4.56f};
  float nozzle_velocity_spread = 0.4f;
  float nozzle_acceleration_spread = 0.0f;
  float nozzle_angle = 0.25f;
  float particle_start_temperature = 20000.0f;
  float particle_end_temperature = 9001.0f;
  // float nozzle_flow_rate = 1000.0f;


};




struct VoxelData {
  bool dirty = false;
  float density;
  float temperature;
};
#pragma pack(push, 1) // Ensure tight packing
struct CompressedVoxel {
  uint8_t x;
  uint8_t y;
  uint8_t z;
  uint8_t density;
  uint8_t temperature;
};
#pragma pack(pop)

// Add helper functions
namespace VoxelCompression {
inline uint8_t compressTemp(float temp) {
  // Scale temperature range (e.g. 300K-10000K) to 0-255
  const float MIN_TEMP = 300.0f;
  const float MAX_TEMP = 1700.0f;
  float normalized = (temp - MIN_TEMP) / (MAX_TEMP - MIN_TEMP);
  return static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, normalized * 255.0f)));
}

inline uint8_t compressDensity(float density) {
  return static_cast<uint8_t>(fminf(255.0f, fmaxf(0.0f, density * 255.0f)));
}
} // namespace VoxelCompression

// Basic data structures that both CPU and GPU code need to know about
struct targetData {
  float origin[3];
  VoxelData *voxels;

  bool owns_voxels = true;
  unsigned int voxel_size;
  float world_size;

  // Constants
  static const float MELTING_POINT;    // K
  static const float HEAT_CAPACITY;    // simplified specific heat
  static const float HEAT_DISSIPATION; // rate of heat transfer between voxels

  targetData();
  targetData(float start_temp);
  targetData(const targetData *other);
  ~targetData();
};

struct particleData {
  bool is_active;
  float position[3];
  float velocity[3];
  float acceleration[3];
  float lifetime;

  particleData();
};

// Forward declaration of the simulation class
class SimulationInstance;

// C-style interface for CUDA functions
#ifdef __cplusplus
extern "C" {
#endif

void setupSimulation(SimulationInstance *sim);
void computeAndRenderSimulation(SimulationInstance *sim, float delta_time);
void teardownSimulation(SimulationInstance *sim);
void cudaTargetToHost(targetData *host_target, SimulationInstance *sim);
void cudaParticlesToHost(particleData *host_particles, SimulationInstance *sim);
void resetCudaData(SimulationInstance *sim);
#ifdef __cplusplus
}
#endif

#endif // KERNEL_H
