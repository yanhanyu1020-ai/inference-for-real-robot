#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>

#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace fs = std::filesystem;
using unitree::robot::ChannelFactory;
using unitree::robot::ChannelSubscriber;
using unitree::robot::ChannelSubscriberPtr;

namespace {

std::atomic<bool> running{true};

void handle_signal(int) {
  running.store(false);
}

int64_t steady_now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int64_t wall_now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

template <typename Range>
void append_values(std::ostream& out, const Range& values) {
  for (const auto& value : values) {
    out << ',' << +value;
  }
}

class Recorder {
 public:
  explicit Recorder(const fs::path& output_dir)
      : output_dir_(output_dir),
        wireless_file_(output_dir / "wireless_controller.csv"),
        sport_file_(output_dir / "sport_mode_state.csv"),
        low_file_(output_dir / "low_state.csv") {
    if (!wireless_file_ || !sport_file_ || !low_file_) {
      throw std::runtime_error("failed to open one or more output files");
    }
    wireless_file_ << std::setprecision(9);
    sport_file_ << std::setprecision(9);
    low_file_ << std::setprecision(9);
    write_headers();
  }

  void start() {
    wireless_sub_.reset(
        new ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>(
            "rt/wirelesscontroller"));
    sport_sub_.reset(
        new ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>(
            "rt/sportmodestate"));
    low_sub_.reset(new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(
        "rt/lowstate"));

    wireless_sub_->InitChannel(
        std::bind(&Recorder::on_wireless, this, std::placeholders::_1), 32);
    sport_sub_->InitChannel(
        std::bind(&Recorder::on_sport, this, std::placeholders::_1), 32);
    low_sub_->InitChannel(
        std::bind(&Recorder::on_low, this, std::placeholders::_1), 32);
  }

  void flush() {
    std::scoped_lock lock(wireless_mutex_, sport_mutex_, low_mutex_);
    wireless_file_.flush();
    sport_file_.flush();
    low_file_.flush();
  }

  uint64_t wireless_count() const { return wireless_count_.load(); }
  uint64_t sport_count() const { return sport_count_.load(); }
  uint64_t low_count() const { return low_count_.load(); }

 private:
  void write_headers() {
    wireless_file_
        << "monotonic_ns,wall_time_ns,seq,lx,ly,rx,ry,keys\n";

    sport_file_
        << "monotonic_ns,wall_time_ns,seq,robot_sec,robot_nanosec,error_code,"
           "mode,gait_type,progress,foot_raise_height,"
           "position_x,position_y,position_z,body_height,"
           "velocity_x,velocity_y,velocity_z,yaw_speed,"
           "quat_w,quat_x,quat_y,quat_z,"
           "gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z,"
           "roll,pitch,yaw,"
           "range_0,range_1,range_2,range_3,"
           "foot_force_0,foot_force_1,foot_force_2,foot_force_3\n";

    low_file_
        << "monotonic_ns,wall_time_ns,seq,tick,power_v,power_a,"
           "quat_w,quat_x,quat_y,quat_z,"
           "gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z,"
           "roll,pitch,yaw,"
           "foot_force_0,foot_force_1,foot_force_2,foot_force_3,"
           "foot_force_est_0,foot_force_est_1,foot_force_est_2,"
           "foot_force_est_3";
    for (int i = 0; i < 40; ++i) {
      low_file_ << ",wireless_raw_" << i;
    }
    for (int i = 0; i < 20; ++i) {
      low_file_ << ",motor_" << i << "_mode"
                << ",motor_" << i << "_q"
                << ",motor_" << i << "_dq"
                << ",motor_" << i << "_ddq"
                << ",motor_" << i << "_tau_est"
                << ",motor_" << i << "_temperature"
                << ",motor_" << i << "_lost";
    }
    low_file_ << '\n';
  }

  void on_wireless(const void* message) {
    const auto& msg =
        *static_cast<const unitree_go::msg::dds_::WirelessController_*>(
            message);
    const auto seq = wireless_count_.fetch_add(1);
    std::lock_guard<std::mutex> lock(wireless_mutex_);
    wireless_file_ << steady_now_ns() << ',' << wall_now_ns() << ',' << seq
                   << ',' << msg.lx() << ',' << msg.ly() << ',' << msg.rx()
                   << ',' << msg.ry() << ',' << msg.keys() << '\n';
    wireless_file_.flush();
    write_stick_json(msg);
  }

  void write_stick_json(
      const unitree_go::msg::dds_::WirelessController_& msg) {
    const fs::path tmp = output_dir_ / ".stick.json.tmp";
    const fs::path dst = output_dir_ / "stick.json";
    {
      std::ofstream out(tmp);
      if (!out) {
        return;
      }
      out << std::setprecision(9) << "{\"lx\":" << msg.lx()
          << ",\"ly\":" << msg.ly() << ",\"rx\":" << msg.rx()
          << ",\"ry\":" << msg.ry() << ",\"keys\":" << msg.keys() << "}\n";
    }
    std::error_code ec;
    fs::rename(tmp, dst, ec);
  }

  void on_sport(const void* message) {
    const auto& msg =
        *static_cast<const unitree_go::msg::dds_::SportModeState_*>(message);
    const auto seq = sport_count_.fetch_add(1);
    const auto& imu = msg.imu_state();
    std::lock_guard<std::mutex> lock(sport_mutex_);
    sport_file_ << steady_now_ns() << ',' << wall_now_ns() << ',' << seq << ','
                << msg.stamp().sec() << ',' << msg.stamp().nanosec() << ','
                << msg.error_code() << ',' << +msg.mode() << ','
                << +msg.gait_type() << ',' << msg.progress() << ','
                << msg.foot_raise_height();
    append_values(sport_file_, msg.position());
    sport_file_ << ',' << msg.body_height();
    append_values(sport_file_, msg.velocity());
    sport_file_ << ',' << msg.yaw_speed();
    append_values(sport_file_, imu.quaternion());
    append_values(sport_file_, imu.gyroscope());
    append_values(sport_file_, imu.accelerometer());
    append_values(sport_file_, imu.rpy());
    append_values(sport_file_, msg.range_obstacle());
    append_values(sport_file_, msg.foot_force());
    sport_file_ << '\n';
  }

  void on_low(const void* message) {
    const auto& msg =
        *static_cast<const unitree_go::msg::dds_::LowState_*>(message);
    const auto seq = low_count_.fetch_add(1);
    const auto& imu = msg.imu_state();
    std::lock_guard<std::mutex> lock(low_mutex_);
    low_file_ << steady_now_ns() << ',' << wall_now_ns() << ',' << seq << ','
              << msg.tick() << ',' << msg.power_v() << ',' << msg.power_a();
    append_values(low_file_, imu.quaternion());
    append_values(low_file_, imu.gyroscope());
    append_values(low_file_, imu.accelerometer());
    append_values(low_file_, imu.rpy());
    append_values(low_file_, msg.foot_force());
    append_values(low_file_, msg.foot_force_est());
    append_values(low_file_, msg.wireless_remote());
    for (const auto& motor : msg.motor_state()) {
      low_file_ << ',' << +motor.mode() << ',' << motor.q() << ','
                << motor.dq() << ',' << motor.ddq() << ',' << motor.tau_est()
                << ',' << +motor.temperature() << ',' << motor.lost();
    }
    low_file_ << '\n';
  }

  fs::path output_dir_;
  std::ofstream wireless_file_;
  std::ofstream sport_file_;
  std::ofstream low_file_;
  std::mutex wireless_mutex_;
  std::mutex sport_mutex_;
  std::mutex low_mutex_;
  std::atomic<uint64_t> wireless_count_{0};
  std::atomic<uint64_t> sport_count_{0};
  std::atomic<uint64_t> low_count_{0};
  ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_>
      wireless_sub_;
  ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_> sport_sub_;
  ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> low_sub_;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << " NETWORK_INTERFACE OUTPUT_DIRECTORY\n";
    return 2;
  }

  const std::string network_interface = argv[1];
  const fs::path output_dir = argv[2];
  fs::create_directories(output_dir);

  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);

  try {
    ChannelFactory::Instance()->Init(0, network_interface);
    Recorder recorder(output_dir);
    recorder.start();

    std::cout << "Recording Go2 telemetry to " << output_dir << '\n'
              << "DDS interface: " << network_interface << '\n'
              << "This process only subscribes; it publishes no commands."
              << std::endl;

    const auto start_wall_ns = wall_now_ns();
    const auto start_steady_ns = steady_now_ns();
    while (running.load()) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      recorder.flush();
      std::cout << "counts wireless=" << recorder.wireless_count()
                << " sport=" << recorder.sport_count()
                << " low=" << recorder.low_count() << std::endl;
    }
    recorder.flush();

    std::ofstream summary(output_dir / "summary.json");
    summary << "{\n"
            << "  \"start_wall_time_ns\": " << start_wall_ns << ",\n"
            << "  \"end_wall_time_ns\": " << wall_now_ns() << ",\n"
            << "  \"start_monotonic_ns\": " << start_steady_ns << ",\n"
            << "  \"end_monotonic_ns\": " << steady_now_ns() << ",\n"
            << "  \"wireless_messages\": " << recorder.wireless_count()
            << ",\n"
            << "  \"sport_messages\": " << recorder.sport_count() << ",\n"
            << "  \"low_state_messages\": " << recorder.low_count() << "\n"
            << "}\n";

    std::cout << "Recorder stopped cleanly." << std::endl;
  } catch (const std::exception& error) {
    std::cerr << "Recorder error: " << error.what() << std::endl;
    return 1;
  }
  return 0;
}
