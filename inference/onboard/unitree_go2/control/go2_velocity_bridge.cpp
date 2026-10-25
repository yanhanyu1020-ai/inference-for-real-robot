// Persistent Go2 velocity bridge: holds the SportClient lease and accepts
// text commands on a unix-domain socket.
//
// Protocol (one command per line):
//   MOVE <vx> <vy> <vyaw>
//   STAND_UP
//   STAND_DOWN
//   STOP
//
// Safety: hard clamp +-1 m/s / +-3.5 rad/s, STOP if no command for 400 ms,
// STOP on exit.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <errno.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>

namespace {

constexpr float kVxLimit = 1.0f;
constexpr float kVyLimit = 1.0f;
constexpr float kVyawLimit = 3.5f;
constexpr int kWatchdogMs = 400;

std::atomic<bool> g_running{true};
std::string g_sock_path;

void on_signal(int) { g_running = false; }

float clamp(float value, float limit) {
  return std::max(-limit, std::min(limit, value));
}

void unlink_sock() {
  if (!g_sock_path.empty()) {
    ::unlink(g_sock_path.c_str());
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: go2_velocity_bridge NETWORK_INTERFACE SOCK_PATH\n";
    return 2;
  }

  const std::string interface_name(argv[1]);
  g_sock_path = argv[2];

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  std::atexit(unlink_sock);

  unitree::robot::ChannelFactory::Instance()->Init(0, interface_name);
  unitree::robot::go2::SportClient client(/*withLease=*/true);
  client.SetTimeout(10.0f);
  client.Init();
  std::cout << "waiting for SportClient lease..." << std::endl;
  client.WaitLeaseApplied();
  std::cout << "lease applied" << std::endl;

  ::unlink(g_sock_path.c_str());
  const int server_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (server_fd < 0) {
    std::cerr << "socket failed: " << std::strerror(errno) << "\n";
    return 1;
  }

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  if (g_sock_path.size() >= sizeof(addr.sun_path)) {
    std::cerr << "sock path too long\n";
    return 1;
  }
  std::strncpy(addr.sun_path, g_sock_path.c_str(), sizeof(addr.sun_path) - 1);
  if (::bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
    std::cerr << "bind failed: " << std::strerror(errno) << "\n";
    return 1;
  }
  if (::listen(server_fd, 8) != 0) {
    std::cerr << "listen failed: " << std::strerror(errno) << "\n";
    return 1;
  }
  const int flags = ::fcntl(server_fd, F_GETFL, 0);
  ::fcntl(server_fd, F_SETFL, flags | O_NONBLOCK);
  std::cout << "listening on " << g_sock_path << std::endl;

  float cmd_vx = 0.0f;
  float cmd_vy = 0.0f;
  float cmd_vyaw = 0.0f;
  bool moving = false;
  auto last_cmd = std::chrono::steady_clock::now();
  std::string pending;

  auto issue_stop = [&]() {
    client.StopMove();
    cmd_vx = cmd_vy = cmd_vyaw = 0.0f;
    moving = false;
  };

  auto handle_line = [&](const std::string &line) {
    std::istringstream iss(line);
    std::string op;
    iss >> op;
    if (op.empty()) {
      return;
    }
    last_cmd = std::chrono::steady_clock::now();
    if (op == "MOVE") {
      float vx = 0.0f, vy = 0.0f, vyaw = 0.0f;
      iss >> vx >> vy >> vyaw;
      cmd_vx = clamp(vx, kVxLimit);
      cmd_vy = clamp(vy, kVyLimit);
      cmd_vyaw = clamp(vyaw, kVyawLimit);
      moving = true;
      const int rc = client.Move(cmd_vx, cmd_vy, cmd_vyaw);
      if (rc != 0) {
        std::cerr << "Move failed rc=" << rc << "\n";
      }
    } else if (op == "STAND_UP") {
      issue_stop();
      const int rc = client.StandUp();
      std::cout << "STAND_UP rc=" << rc << std::endl;
    } else if (op == "STAND_DOWN") {
      issue_stop();
      const int rc = client.StandDown();
      std::cout << "STAND_DOWN rc=" << rc << std::endl;
    } else if (op == "STOP") {
      issue_stop();
      std::cout << "STOP" << std::endl;
    } else {
      std::cerr << "unknown command: " << op << "\n";
    }
  };

  while (g_running) {
    const int client_fd = ::accept(server_fd, nullptr, nullptr);
    if (client_fd >= 0) {
      char buffer[512];
      pending.clear();
      while (true) {
        const ssize_t n = ::read(client_fd, buffer, sizeof(buffer));
        if (n <= 0) {
          break;
        }
        pending.append(buffer, buffer + n);
        std::size_t pos;
        while ((pos = pending.find('\n')) != std::string::npos) {
          handle_line(pending.substr(0, pos));
          pending.erase(0, pos + 1);
        }
      }
      if (!pending.empty()) {
        handle_line(pending);
      }
      ::close(client_fd);
    } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
      std::cerr << "accept failed: " << std::strerror(errno) << "\n";
    }

    const auto now = std::chrono::steady_clock::now();
    const auto idle_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_cmd).count();
    if (moving && idle_ms > kWatchdogMs) {
      std::cerr << "watchdog: no command for " << idle_ms << " ms, STOP\n";
      issue_stop();
    } else if (moving) {
      // Keep streaming the last velocity so the dog does not coast to a halt.
      client.Move(cmd_vx, cmd_vy, cmd_vyaw);
      last_cmd = now;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  std::cout << "shutting down" << std::endl;
  ::close(server_fd);
  unlink_sock();
  // StopMove() / SportClient dtor during teardown trips
  // "RecurrentThread Execute Func error" / abort.  That can leave the
  // sport lease held so the remote cannot stand_down.  _Exit skips the
  // destructor; the robot drops the lease when this DDS participant dies.
  std::_Exit(0);
}
