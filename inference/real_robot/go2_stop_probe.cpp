// One acknowledged stop request; never issues velocity or posture commands.
#include <exception>
#include <iostream>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: v3_go2_stop_probe NETWORK_INTERFACE\n";
    return 2;
  }
  try {
    unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
    // Do not acquire a control lease or replace an existing controller.
    unitree::robot::go2::SportClient client(false);
    client.SetTimeout(3.0f);
    client.Init();
    std::cout << "sending StopMove once" << std::endl;
    const int rc = client.StopMove();
    std::cout << "StopMove return=" << rc << std::endl;
    return rc == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
