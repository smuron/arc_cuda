#include "server.h"
#include "kernel.h"
#include "simulation.h"
#include <chrono>
#include <random>
#include <stdio.h>
#include <thread>
#include <unistd.h>


int main() {

  auto startTime = std::chrono::steady_clock::now();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  

  SimulationInstance sim(1000); // Start with 1000 particles
  VisualizationServer visServer;
  if (!visServer.init()) {
    printf("Failed to initialize visualization server\n");
    return -1;
  }
  sim.visServer = &visServer;
  visServer.sim = &sim;
  setupSimulation(&sim);

  sim.startNetworking();

  double lastTime = 0.0;
  int frameCount = 0;
  double lastFPSDisplay = 0.0;
  double frameTime = 0.02;

  // Render loop
  
  while (true) {
    float currentTime = std::chrono::duration<float>(
                            std::chrono::steady_clock::now() - startTime)
                            .count();
    float delta_time = currentTime - lastTime;
    if (delta_time < frameTime) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }
    computeAndRenderSimulation(&sim, delta_time);

    frameCount++;
    if (currentTime - lastFPSDisplay >= 7.0) {
      double fps = frameCount / (currentTime - lastFPSDisplay);
      printf("FPS: %.1f\n", fps);
      frameCount = 0;
      lastFPSDisplay = currentTime;
    }

    lastTime = currentTime;
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }

  sim.stopNetworking();
  teardownSimulation(&sim);

  return 0;
}
