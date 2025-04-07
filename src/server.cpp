#include "server.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <unistd.h>

#include "json.hpp"
#include "kernel.h"
#include <libwebsockets.h>

using json = nlohmann::json;

static VisualizationServer *g_server = nullptr;

std::string generateUUID() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 15);
  std::stringstream ss;
  ss << std::hex;
  for (int i = 0; i < 8; i++)
    ss << dis(gen);
  ss << "-";
  for (int i = 0; i < 4; i++)
    ss << dis(gen);
  ss << "-4";
  for (int i = 0; i < 3; i++)
    ss << dis(gen);
  ss << "-";
  ss << dis(gen) % 4 + 8;
  for (int i = 0; i < 3; i++)
    ss << dis(gen);
  ss << "-";
  for (int i = 0; i < 12; i++)
    ss << dis(gen);
  return ss.str();
}

bool VisualizationServer::init() { return initWebRTC(); }

bool VisualizationServer::initWebRTC() {
  struct lws_context_creation_info info;
  memset(&info, 0, sizeof(info));
  info.port = 8080;
  info.protocols = new lws_protocols[2]{
      {"webrtc-signaling", &VisualizationServer::wsCallback, 0, 4096},
      {nullptr, nullptr, 0, 0}};

  g_server = this;
  ws_context = lws_create_context(&info);
  if (!ws_context) {
    std::cerr << "WebSocket context creation failed" << std::endl;
    return false;
  }
  std::cout << "WebSocket server started on port " << info.port << std::endl;

  // Use a joinable thread instead of detaching
  service_thread = std::thread([this]() {
    std::cout << "WebSocket service thread started" << std::endl;
    while (ws_context && running) {
      lws_service(ws_context, 50);
    }
    std::cout << "WebSocket service thread stopped" << std::endl;
  });

  return true;
}

int VisualizationServer::wsCallback(struct lws *wsi,
                                    enum lws_callback_reasons reason,
                                    void *user, void *in, size_t len) {
  if (!g_server) {
    std::cerr << "wsCallback: g_server is null" << std::endl;
    return -1;
  }
  if (!g_server->running) {
    std::cout << "Server shutting down, ignoring callback" << std::endl;
    return -1;
  }

  switch (reason) {
  case LWS_CALLBACK_ESTABLISHED: {
    if (!wsi) {
      std::cerr << "wsCallback: wsi is null for ESTABLISHED" << std::endl;
      return -1;
    }
    std::string peer_id = generateUUID();
    json msg = {{"type", "peer_id"}, {"id", peer_id}};
    std::string msg_str = msg.dump();
    std::cout << "Generated peer_id: " << peer_id << std::endl;
    {
      std::lock_guard<std::mutex> lock(g_server->rtc_peers_mutex);
      std::cout << "Constructing peer: " << peer_id << std::endl;
      WebRTCPeer peer(peer_id, wsi, nullptr, nullptr,
                      msg_str); // Explicit construction
      std::cout << "Inserting peer: " << peer_id << std::endl;
      auto [it, inserted] =
          g_server->rtc_peers.insert({peer_id, std::move(peer)});
      if (!inserted) {
        std::cerr << "Failed to insert peer: " << peer_id
                  << " (already exists?)" << std::endl;
        return -1;
      }
      std::cout << "Insert complete for peer: " << peer_id << std::endl;
      // Use iterator to access the inserted peer, avoiding operator[]
      std::string &pending = *it->second.pending_message;
      size_t len = pending.size();

      // Allocate buffer with LWS_PRE padding
      std::vector<unsigned char> buf(LWS_PRE + len);
      // Copy data into buffer after LWS_PRE
      memcpy(buf.data() + LWS_PRE, pending.data(), len);

      // Write with padding
      int bytes_written =
          lws_write(wsi,
                    buf.data() + LWS_PRE, // Pointer to data, after LWS_PRE
                    len, LWS_WRITE_TEXT);
      if (bytes_written <
          static_cast<int>(len)) { // Check per docs recommendation
        std::cerr << "Failed to send peer_id: " << bytes_written << std::endl;
        return -1;
      }
    }
    std::cout << "Sent peer_id to client: " << peer_id << std::endl;
    break;
  }

  case LWS_CALLBACK_RECEIVE: {
    if (!wsi || !in || len == 0) {
      std::cerr << "wsCallback: Invalid RECEIVE parameters" << std::endl;
      return -1;
    }
    std::string msg((char *)in, len);
    try {
      json j = json::parse(msg);
      std::string peer_id = j["peer_id"];
      std::lock_guard<std::mutex> lock(g_server->rtc_peers_mutex);
      auto it = g_server->rtc_peers.find(peer_id);
      if (it != g_server->rtc_peers.end()) {
        g_server->handleWebRTCSignaling(it->second, msg);
      }
    } catch (const std::exception &e) {
      std::cerr << "Error parsing WebSocket message: " << e.what() << std::endl;
      std::cerr << msg << std::endl;
    }
    break;
  }

  case LWS_CALLBACK_CLOSED: {
    if (!wsi) {
      std::cerr << "wsCallback: wsi is null for CLOSED" << std::endl;
      return -1;
    }
    std::lock_guard<std::mutex> lock(g_server->rtc_peers_mutex);
    for (auto it = g_server->rtc_peers.begin();
         it != g_server->rtc_peers.end();) {
      if (it->second.wsi == wsi) {
        std::cout << "Peer " << *(it->second.id)
                  << " disconnecting, wsi = " << wsi << std::endl;
        it->second.cleanup();
        std::cout << "Erasing peer: " << *(it->second.id) << std::endl;
        it = g_server->rtc_peers.erase(it);
        std::cout << "Removed peer on close" << std::endl;
      } else {
        ++it;
      }
    }
    break;
  }
  }
  return 0;
}

void VisualizationServer::handleWebRTCSignaling(WebRTCPeer &peer,
                                                const std::string &message) {
  json j = json::parse(message);
  std::string type = j["type"];
  std::cout << "Handling signaling message type: " << type << std::endl;

  if (type == "offer") {
    rtc::Configuration config;
    config.iceServers = {{"stun:stun.l.google.com:19302"}};
    // Leave disableAutoNegotiation as false (default)
    peer.peer_connection = std::make_shared<rtc::PeerConnection>(config);
    std::string sdp = j["sdp"];
    std::cout << "Received SDP:\n" << sdp << std::endl;

    peer.data_channel = peer.peer_connection->createDataChannel(
        "simulation_data",
        rtc::DataChannelInit{
            .reliability = {
                .unordered = true,
                .maxPacketLifeTime = std::chrono::milliseconds(100),
            }});

    // Set up callbacks before setting remote description
    peer.peer_connection->onLocalCandidate([&peer](rtc::Candidate candidate) {
      json ice_msg = {{"type", "ice_candidate"},
                      {"candidate", candidate.candidate()},
                      {"peer_id", *peer.id}};
      std::string ice_str = ice_msg.dump();
      std::cout << "Sending ICE candidate: " << ice_str << std::endl;

      size_t len = ice_str.size();
      std::vector<unsigned char> buf(LWS_PRE + len);
      memcpy(buf.data() + LWS_PRE, ice_str.data(), len);
      int bytes_written =
          lws_write(peer.wsi, buf.data() + LWS_PRE, len, LWS_WRITE_TEXT);
      if (bytes_written < static_cast<int>(len)) {
        std::cerr << "Failed to send ICE candidate: " << bytes_written << " of "
                  << len << std::endl;
      } else {
        std::cout << "Sent ICE candidate, " << bytes_written << " bytes"
                  << std::endl;
      }
    });

    peer.peer_connection->onLocalDescription([&peer](
                                                 const rtc::Description &desc) {
      if (desc.type() == rtc::Description::Type::Answer) {
        json answer = {{"type", "answer"}, {"sdp", std::string(desc)}};
        std::string answer_str = answer.dump();
        std::cout << "Sending answer from onLocalDescription: " << answer_str
                  << std::endl;

        size_t answer_len = answer_str.size();
        std::vector<unsigned char> answer_buf(LWS_PRE + answer_len);
        memcpy(answer_buf.data() + LWS_PRE, answer_str.data(), answer_len);
        int bytes_written = lws_write(peer.wsi, answer_buf.data() + LWS_PRE,
                                      answer_len, LWS_WRITE_TEXT);
        if (bytes_written < static_cast<int>(answer_len)) {
          std::cerr << "Failed to send answer, wrote " << bytes_written
                    << " of " << answer_len << std::endl;
        } else {
          std::cout << "Answer sent successfully, " << bytes_written << " bytes"
                    << std::endl;
        }
      }
    });

    try {
      rtc::Description offer(sdp, "offer");
      peer.peer_connection->setRemoteDescription(offer);
      std::cout
          << "Remote description (offer) set successfully, signaling state: "
          << static_cast<int>(peer.peer_connection->signalingState())
          << std::endl;
      // Auto-negotiation should trigger onLocalDescription with the answer
    } catch (const std::exception &e) {
      std::cerr << "Failed to set remote description: " << e.what()
                << std::endl;
      return;
    }

    peer.data_channel->onOpen([]() { std::cout << "Data channel opened\n"; });
    peer.data_channel->onMessage(
        [&peer, this](const std::variant<std::string, rtc::binary> &msg) {
          if (std::holds_alternative<rtc::binary>(msg)) {
            const rtc::binary &binary_data = std::get<rtc::binary>(msg);
            if (binary_data.size() < sizeof(PacketHeader)) {
              std::cerr << "Received invalid packet: too small" << std::endl;
              return;
            }

            const PacketHeader *header =
                reinterpret_cast<const PacketHeader *>(binary_data.data());
            switch (header->type) {
            case PACKET_PARTICLES:
              // Already handled elsewhere; could log or validate here if needed
              break;
            case PACKET_VOXELS:
              // Already handled elsewhere
              break;
            case PACKET_CONTROL:
              handleControlMessage(peer, binary_data);
              break;
            default:
              std::cerr << "Unknown packet type: " << header->type << std::endl;
            }
          } else {
            std::cerr << "Received unexpected string message on data channel"
                      << std::endl;
          }
        });
  } else if (type == "ice_candidate") {
    std::string candidate = j["candidate"];
    std::cout << "Adding ICE candidate: " << candidate << std::endl;
    peer.peer_connection->addRemoteCandidate(rtc::Candidate(candidate));
  }
}

void VisualizationServer::handleControlMessage(WebRTCPeer& peer, const rtc::binary& data) {
    const PacketHeader* header = reinterpret_cast<const PacketHeader*>(data.data());
    if (data.size() < sizeof(PacketHeader)) {
        std::cerr << "Control packet too small" << std::endl;
        return;
    }

    // Extract the JSON payload after the header
    const char* json_data = reinterpret_cast<const char*>(data.data() + sizeof(PacketHeader));
    size_t json_len = data.size() - sizeof(PacketHeader);

    try {
        json j = json::parse(std::string(json_data, json_len));
        std::cout << "Received control message from peer " << *peer.id << ": " << j.dump() << std::endl;

        // Example: Handle specific parameters
        if (j.contains("parameters")) {
            SimParameters params = j["parameters"];
            std::cout << "Control parameters: " << params.dump() << std::endl;
            sim->updateParameters(params);
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse control message: " << e.what() << std::endl;
    }
}

void VisualizationServer::broadcastParticleData(particleData *particles,
                                                int count) {
  std::lock_guard<std::mutex> lock(rtc_peers_mutex);
  for (auto &[id, peer] : rtc_peers) {
    if (peer.data_channel && peer.data_channel->isOpen()) {
      sendParticleDataToChannel(particles, count, peer.data_channel);
    }
  }
}

void VisualizationServer::broadcastVoxelData(targetData *target) {
  std::lock_guard<std::mutex> lock(rtc_peers_mutex);
  for (auto &[id, peer] : rtc_peers) {
    if (peer.data_channel && peer.data_channel->isOpen()) {
      sendVoxelDataToChannel(target, peer.data_channel);
    }
  }
}

void VisualizationServer::sendParticleDataToChannel(
    particleData *particles, int count,
    std::shared_ptr<rtc::DataChannel> channel) {
  size_t packet_size = sizeof(PacketHeader) + (count * 3 * sizeof(float) * 2);
  std::vector<char> buffer(packet_size);

  PacketHeader *header = (PacketHeader *)buffer.data();
  header->type = PACKET_PARTICLES;
  header->size = count;
  header->total_elements = count;

  float *data_ptr = (float *)(buffer.data() + sizeof(PacketHeader));
  for (int i = 0; i < count; i++) {
    memcpy(data_ptr + (i * 3), particles[i].position, sizeof(float) * 3);
  }
  float *vel_ptr = data_ptr + (count * 3);
  for (int i = 0; i < count; i++) {
    memcpy(vel_ptr + (i * 3), particles[i].velocity, sizeof(float) * 3);
  }

  if (channel && channel->isOpen()) {
    channel->send(reinterpret_cast<const std::byte *>(buffer.data()),
                  packet_size);
  }
}

void VisualizationServer::sendVoxelDataToChannel(
    targetData *target, std::shared_ptr<rtc::DataChannel> channel) {
  const size_t grid_size = target->voxel_size;
  std::vector<CompressedVoxel> exposed_voxels;
  exposed_voxels.reserve(grid_size * grid_size * 6);

  for (size_t z = 0; z < grid_size; z++) {
    for (size_t y = 0; y < grid_size; y++) {
      for (size_t x = 0; x < grid_size; x++) {
        size_t idx = z * grid_size * grid_size + y * grid_size + x;
        if (!target->voxels[idx].dirty)
          continue;

        bool is_exposed = false;
        const int neighbors[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                     {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
        for (int n = 0; n < 6; n++) {
          int nx = x + neighbors[n][0];
          int ny = y + neighbors[n][1];
          int nz = z + neighbors[n][2];
          if (nx < 0 || nx >= grid_size || ny < 0 || ny >= grid_size ||
              nz < 0 || nz >= grid_size) {
            is_exposed = true;
            break;
          }
          size_t nidx = nz * grid_size * grid_size + ny * grid_size + nx;
          if (target->voxels[nidx].density <= 0.0f) {
            is_exposed = true;
            break;
          }
        }
        if (is_exposed) {
          CompressedVoxel cv;
          cv.x = x;
          cv.y = y;
          cv.z = z;
          cv.density =
              VoxelCompression::compressDensity(target->voxels[idx].density);
          cv.temperature =
              VoxelCompression::compressTemp(target->voxels[idx].temperature);
          exposed_voxels.push_back(cv);
        }
      }
    }
  }

  const size_t MAX_VOXELS_PER_PACKET = 8192;
  const size_t packets_needed =
      (exposed_voxels.size() + MAX_VOXELS_PER_PACKET - 1) /
      MAX_VOXELS_PER_PACKET;

  for (size_t packet = 0; packet < packets_needed; packet++) {
    size_t start_voxel = packet * MAX_VOXELS_PER_PACKET;
    size_t voxels_this_packet =
        std::min(MAX_VOXELS_PER_PACKET, exposed_voxels.size() - start_voxel);
    size_t packet_size = sizeof(PacketHeader) + (voxels_this_packet * 5);

    std::vector<char> buffer(packet_size);
    PacketHeader *header = (PacketHeader *)buffer.data();
    header->type = PACKET_VOXELS;
    header->size = grid_size;
    header->total_elements = voxels_this_packet;

    uint8_t *write_ptr = (uint8_t *)(buffer.data() + sizeof(PacketHeader));
    for (size_t i = 0; i < voxels_this_packet; i++) {
      const auto &voxel = exposed_voxels[start_voxel + i];
      *write_ptr++ = voxel.x;
      *write_ptr++ = voxel.y;
      *write_ptr++ = voxel.z;
      *write_ptr++ = voxel.density;
      *write_ptr++ = voxel.temperature;
    }

    if (channel && channel->isOpen()) {
      channel->send(reinterpret_cast<const std::byte *>(buffer.data()),
                    packet_size);
    }
    usleep(1000);
  }
}

VisualizationServer::~VisualizationServer() {
  running = false; // Signal thread to stop
  if (service_thread.joinable()) {
    service_thread.join(); // Wait for thread to finish
  }
  if (ws_context) {
    lws_context_destroy(ws_context);
    ws_context = nullptr;
  }
}
