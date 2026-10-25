// Read-only mode/service inspection. This binary contains no motion commands.
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/go2/robot_state/robot_state_client.hpp>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: v3_go2_inspect NETWORK_INTERFACE\n";
    return 2;
  }
  unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
  unitree::robot::b2::MotionSwitcherClient mode;
  mode.SetTimeout(3.0f);
  mode.Init();
  std::string form, name;
  const int mode_rc = mode.CheckMode(form, name);
  std::cout << "CheckMode return=" << mode_rc << " form=" << std::quoted(form)
            << " name=" << std::quoted(name) << '\n';

  unitree::robot::go2::RobotStateClient robot;
  robot.SetTimeout(3.0f);
  robot.Init();
  std::vector<unitree::robot::go2::ServiceState> services;
  const int service_rc = robot.ServiceList(services);
  std::cout << "ServiceList return=" << service_rc << " count=" << services.size() << '\n';
  for (const auto& service : services) {
    std::cout << "service=" << std::quoted(service.name)
              << " status=" << service.status << " protect=" << service.protect << '\n';
  }
  return mode_rc == 0 && service_rc == 0 ? 0 : 1;
}
