#include "server.h"
#include "kernel.h"  // for particleData and targetData types

#include <iostream>
#include <cmath>     // for sin, cos, M_PI
#include <unistd.h>  // for sleep
#include <cstring>   // for memcpy


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main() {
    VisualizationServer server;
    if (!server.init()) {
        std::cerr << "Failed to initialize server" << std::endl;
        return 1;
    }

    server.startConnectionHandler();  // Start the connection handler thread
    // Create some dummy data
    const int NUM_PARTICLES = 10;
    particleData testParticles[NUM_PARTICLES];
    for (int i = 0; i < NUM_PARTICLES; i++) {
        // Make particles move in a circle
        testParticles[i].position[0] = cos(i * 2 * M_PI / NUM_PARTICLES);
        testParticles[i].position[1] = sin(i * 2 * M_PI / NUM_PARTICLES);
        testParticles[i].position[2] = 0;
        
        testParticles[i].velocity[0] = -sin(i * 2 * M_PI / NUM_PARTICLES);
        testParticles[i].velocity[1] = cos(i * 2 * M_PI / NUM_PARTICLES);
        testParticles[i].velocity[2] = 0;
    }

    // Test voxel data too
    targetData testVoxels;
    testVoxels.voxel_size = 4;  // small 4x4x4 grid
    size_t total_voxels = testVoxels.voxel_size * testVoxels.voxel_size * testVoxels.voxel_size;
    testVoxels.voxels = new VoxelData[total_voxels];
    
    // Fill with some pattern
    for (size_t i = 0; i < total_voxels; i++) {
        testVoxels.voxels[i].density = sin(i * 0.1f);
        testVoxels.voxels[i].temperature = cos(i * 0.1f);
    }

    // Send data in a loop
    while (true) {
        server.broadcastParticleData(testParticles, NUM_PARTICLES);
        server.broadcastVoxelData(&testVoxels);
        // std::cout << "Sent data packet" << std::endl;
        sleep(2);  // Send every second
    }

    delete[] testVoxels.voxels;
    return 0;
}
