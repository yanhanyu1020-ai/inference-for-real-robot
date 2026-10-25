#include <iostream>
#include <stdexcept>
#include <string>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: go2_command_executor NETWORK_INTERFACE COMMAND\n";
    return 2;
  }

  const std::string interface_name(argv[1]);
  const std::string command(argv[2]);
  unitree::robot::ChannelFactory::Instance()->Init(0, interface_name);

  unitree::robot::go2::SportClient client;
  client.SetTimeout(10.0f);
  client.Init();

  int result = 0;
  if (command == "stand_up") {
    result = client.StandUp();
  } else if (command == "stand_down") {
    result = client.StandDown();
  } else if (command == "stop") {
    result = client.StopMove();
  } else if (command == "move_forward") {
    result = client.Move(0.3f, 0.0f, 0.0f);
  } else if (command == "move_backward") {
    result = client.Move(-0.2f, 0.0f, 0.0f);
  } else if (command == "move_left") {
    result = client.Move(0.0f, 0.2f, 0.0f);
  } else if (command == "move_right") {
    result = client.Move(0.0f, -0.2f, 0.0f);
  } else if (command == "turn_left") {
    result = client.Move(0.0f, 0.0f, 0.5f);
  } else if (command == "turn_right") {
    result = client.Move(0.0f, 0.0f, -0.5f);
  } else {
    std::cerr << "unsupported Go2 command: " << command << "\n";
    return 2;
  }

  if (result != 0) {
    std::cerr << "Unitree SDK command failed with code " << result << "\n";
    return 1;
  }
  std::cout << "Go2 command accepted: " << command << "\n";
  return 0;
}
