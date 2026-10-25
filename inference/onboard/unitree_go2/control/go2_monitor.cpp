// Go2 control-signal monitor.
//
// Subscribes to rt/wirelesscontroller (remote raw inputs) and
// rt/sportmodestate (robot state) and prints a compact line ~10 Hz so an
// operator can see what the dog is doing and what the remote sends.
// This process only subscribes; it never publishes.

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

using unitree::robot::ChannelFactory;
using unitree::robot::ChannelSubscriber;
using unitree::robot::ChannelSubscriberPtr;

namespace {

std::atomic<bool> running{true};

struct Snapshot {
  std::atomic<uint64_t> seq{0};
  std::atomic<int> mode{-1};
  std::atomic<int> gait{-1};
  std::atomic<float> vx{0};
  std::atomic<float> vy{0};
  std::atomic<float> vyaw{0};
  std::atomic<float> px{0};
  std::atomic<float> py{0};
  std::atomic<float> height{0};
  std::atomic<int> lx{0};
  std::atomic<int> ly{0};
  std::atomic<int> rx{0};
  std::atomic<int> ry{0};
  std::atomic<uint16_t> keys{0};
  std::atomic<bool> have_wireless{false};
};

Snapshot snap;

void on_wireless(const void* message) {
  const auto& msg =
      *static_cast<const unitree_go::msg::dds_::WirelessController_*>(message);
  snap.lx.store(msg.lx());
  snap.ly.store(msg.ly());
  snap.rx.store(msg.rx());
  snap.ry.store(msg.ry());
  snap.keys.store(msg.keys());
  snap.have_wireless.store(true);
}

void on_sport(const void* message) {
  const auto& msg =
      *static_cast<const unitree_go::msg::dds_::SportModeState_*>(message);
  snap.seq.fetch_add(1);
  snap.mode.store(msg.mode());
  snap.gait.store(msg.gait_type());
  const auto& v = msg.velocity();
  snap.vx.store(v[0]);
  snap.vy.store(v[1]);
  snap.vyaw.store(msg.yaw_speed());
  const auto& p = msg.position();
  snap.px.store(p[0]);
  snap.py.store(p[1]);
  snap.height.store(msg.body_height());
}

void handle_signal(int) { running.store(false); }

const char* mode_name(int mode) {
  switch (mode) {
    case 0: return "IDLE";
    case 1: return "STAND";
    case 2: return "RUN";
    case 3: return "DOWN";
    default: return "?";
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: go2_monitor NETWORK_INTERFACE\n");
    return 2;
  }
  const std::string network_interface = argv[1];

  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);

  ChannelFactory::Instance()->Init(0, network_interface);
  ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_>
      wireless_sub(new ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>(
          "rt/wirelesscontroller"));
  ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_>
      sport_sub(new ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>(
          "rt/sportmodestate"));
  wireless_sub->InitChannel(on_wireless, 16);
  sport_sub->InitChannel(on_sport, 16);

  std::printf("go2 monitor ready on %s\n", network_interface.c_str());
  std::fflush(stdout);

  int last_mode = -1;
  while (running.load()) {
    int mode = snap.mode.load();
    if (mode != last_mode) {
      std::printf(">>> mode -> %s (%d)\n", mode_name(mode), mode);
      std::fflush(stdout);
      last_mode = mode;
    }
    std::printf(
        "seq=%llu mode=%s gait=%d v=[%.2f %.2f %.2f] pos=[%.1f %.1f] h=%.2f"
        " | remote lx=%d ly=%d rx=%d ry=%d keys=%u\n",
        (unsigned long long)snap.seq.load(), mode_name(snap.mode.load()),
        snap.gait.load(), snap.vx.load(), snap.vy.load(), snap.vyaw.load(),
        snap.px.load(), snap.py.load(), snap.height.load(), snap.lx.load(),
        snap.ly.load(), snap.rx.load(), snap.ry.load(),
        (unsigned)snap.keys.load());
    std::fflush(stdout);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return 0;
}
