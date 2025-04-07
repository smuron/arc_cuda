// server.h
#ifndef SERVER_H
#define SERVER_H

#include "kernel.h" // for particleData and targetData types
#include <algorithm>
#include <iostream>
#include <libwebsockets.h> // We'll need this for WebSocket support
#include <mutex>
#include <netinet/in.h>
#include <rtc/rtc.hpp>
#include <sys/socket.h>
#include <thread>
#include <vector>

struct WebRTCPeer {
  std::unique_ptr<std::string> id; // Explicitly managed
  struct lws *wsi;
  std::shared_ptr<rtc::PeerConnection> peer_connection;
  std::shared_ptr<rtc::DataChannel> data_channel;
  std::unique_ptr<std::string> pending_message; // Explicitly managed
  bool cleaned_up = false;

  WebRTCPeer() = default;

  WebRTCPeer(std::string id_, struct lws *wsi_,
             std::shared_ptr<rtc::PeerConnection> pc_,
             std::shared_ptr<rtc::DataChannel> dc_, std::string msg_)
      : id(std::make_unique<std::string>(std::move(id_))), wsi(wsi_),
        peer_connection(std::move(pc_)), data_channel(std::move(dc_)),
        pending_message(std::make_unique<std::string>(std::move(msg_))) {
    std::cout << "Constructing WebRTCPeer: " << *id << std::endl;
    std::cout << "id ptr: " << id.get() << ", str: " << id->c_str() << std::endl;
  std::cout << "pending_message ptr: " << pending_message.get() << ", str: " << pending_message->c_str() << std::endl;
  }

  WebRTCPeer(WebRTCPeer &&other) noexcept
      : id(std::move(other.id)), wsi(other.wsi),
        peer_connection(std::move(other.peer_connection)),
        data_channel(std::move(other.data_channel)),
        pending_message(std::move(other.pending_message)),
        cleaned_up(other.cleaned_up) {
    other.wsi = nullptr;
    std::cout << "Moving WebRTCPeer: " << (id ? *id : "null") << std::endl;
  }

  WebRTCPeer &operator=(WebRTCPeer &&other) noexcept {
    if (this != &other) {
      id = std::move(other.id);
      wsi = other.wsi;
      peer_connection = std::move(other.peer_connection);
      data_channel = std::move(other.data_channel);
      pending_message = std::move(other.pending_message);
      cleaned_up = other.cleaned_up;
      other.wsi = nullptr;
      std::cout << "Move assigning WebRTCPeer: " << (id ? *id : "null")
                << std::endl;
    }
    return *this;
  }

  ~WebRTCPeer() {
    std::cout << "Destroying WebRTCPeer: " << (id ? *id : "null") << std::endl;
    std::cout << "id ptr: " << id.get() << ", str: " << (id ? id->c_str() : "null") << std::endl;
  std::cout << "pending_message ptr: " << pending_message.get() << ", str: " << (pending_message ? pending_message->c_str() : "null") << std::endl;
    std::cout << "  id: " << (id && !id->empty() ? *id : "empty") << std::endl;
    std::cout << "  wsi: " << wsi << std::endl;
    std::cout << "  peer_connection: " << (peer_connection ? "set" : "null")
              << std::endl;
    std::cout << "  data_channel: " << (data_channel ? "set" : "null")
              << std::endl;
    std::cout << "  pending_message: "
              << (pending_message && !pending_message->empty()
                      ? *pending_message
                      : "empty")
              << std::endl;
  }

  void cleanup() {
    if (cleaned_up) {
      std::cout << "Peer " << (id ? *id : "null")
                << " already cleaned up, skipping" << std::endl;
      return;
    }
    std::cout << "Cleaning up WebRTCPeer: " << (id ? *id : "null") << std::endl;
    if (data_channel) {
      if (data_channel->isOpen()) {
        std::cout << "Closing data_channel for peer: " << (id ? *id : "null")
                  << std::endl;
        data_channel->close();
      }
      data_channel.reset();
    }
    if (peer_connection) {
      if (peer_connection->state() != rtc::PeerConnection::State::Closed) {
        std::cout << "Closing peer_connection for peer: " << (id ? *id : "null")
                  << std::endl;
        peer_connection->close();
      }
      peer_connection.reset();
    }
    cleaned_up = true;
    std::cout << "Cleanup complete for peer: " << (id ? *id : "null")
              << ", wsi = " << wsi << std::endl;
  }
};

enum PacketType { PACKET_PARTICLES = 1, PACKET_VOXELS = 2, PACKET_CONTROL = 3 };

#pragma pack(push, 1) // Force 1-byte alignment
struct PacketHeader {
  uint32_t type;           // PacketType enum
  uint32_t size;           // grid size for voxels, particle count for particles
  uint32_t total_elements; // total number of elements following the header
};
#pragma pack(pop)

struct VisualizationServer {
  struct ParticlePacket {
    uint32_t num_particles;
    struct {
      float pos[3];
      float vel[3];
    } particles[];
  };

  struct VoxelPacket {
    uint32_t size;
    VoxelData data[]; // density,temp?
  };

  struct lws_context *ws_context;
  std::map<std::string, WebRTCPeer> rtc_peers;
  std::mutex rtc_peers_mutex;
  std::thread service_thread;
  bool running = true;
  SimulationInstance *sim;

  // New methods
  bool init();
  bool initWebRTC();
  static int wsCallback(struct lws *wsi, enum lws_callback_reasons reason,
                        void *user, void *in, size_t len);
  void handleWebRTCSignaling(WebRTCPeer &peer, const std::string &message);

  void handleControlMessage(WebRTCPeer& peer, const std::string& data);
  void broadcastParticleData(particleData *particles, int count);
  void broadcastVoxelData(targetData *target);

  // Updated methods
  void sendParticleDataToChannel(particleData *particles, int count,
                                 std::shared_ptr<rtc::DataChannel> channel);
  void sendVoxelDataToChannel(targetData *target,
                              std::shared_ptr<rtc::DataChannel> channel);

  ~VisualizationServer();
};

#endif // SERVER_H
