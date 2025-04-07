#ifndef SIMULATION_H
#define SIMULATION_H

#include "kernel.h"
#include <mutex>
#include <thread>

#include "json.hpp"

class VisualizationServer;
class SimulationInstance {
public:
  // Host data
  targetData *target;
  particleData *particles;

  // Simulation parameters
  SimParameters params;

  unsigned int num_particles;

  SimulationInstance(unsigned int particle_count = 1000);
  ~SimulationInstance();

  // Add double-buffered CPU copies for networking
  struct NetworkBuffer {
    targetData *target;
    particleData *particles;
    std::mutex mutex;
    bool ready;
  };
  NetworkBuffer network_buffers[2];
  int current_network_buffer;

  VisualizationServer *visServer; // Add this line

  // Network thread
  std::thread network_thread;
  bool network_running;
  float network_update_rate; // How often to send updates

  void networkThreadFunc();
  void startNetworking();
  void stopNetworking();

  void updateParameters(const SimParameters &params);
  void resetAll();

private:
  // CUDA-specific members can be hidden in the implementation
  struct CUDAData;
  CUDAData *cuda_data;

  friend void setupSimulation(SimulationInstance *sim);
  friend void computeAndRenderSimulation(SimulationInstance *sim,
                                         float delta_time);
  friend void teardownSimulation(SimulationInstance *sim);
  friend void cudaTargetToHost(targetData *host_target,
                               SimulationInstance *sim);
  friend void cudaParticlesToHost(particleData *host_particles,
                                  SimulationInstance *sim);
  friend void resetCudaData(SimulationInstance *sim);
};
namespace nlohmann {
void to_json(nlohmann::json &j, const SimParameters &p);
void from_json(const nlohmann::json &j, SimParameters &p);
} // namespace nlohmann
inline std::ostream &operator<<(std::ostream &os, const SimParameters &p) {
  os << "SimParameters{\n"
     << "  nozzle_origin: [" << p.nozzle_origin[0] << ", " << p.nozzle_origin[1]
     << ", " << p.nozzle_origin[2] << "],\n"
     << "  nozzle_initial_acceleration: [" << p.nozzle_initial_acceleration[0]
     << ", " << p.nozzle_initial_acceleration[1] << ", "
     << p.nozzle_initial_acceleration[2] << "],\n"
     << "  nozzle_initial_velocity: [" << p.nozzle_initial_velocity[0] << ", "
     << p.nozzle_initial_velocity[1] << ", " << p.nozzle_initial_velocity[2]
     << "],\n"
     << "  nozzle_velocity_spread: " << p.nozzle_velocity_spread << ",\n"
     << "  nozzle_acceleration_spread: " << p.nozzle_acceleration_spread
     << ",\n"
     << "  nozzle_angle: " << p.nozzle_angle << ",\n";
  return os;
}
#endif // SIMULATION_H
