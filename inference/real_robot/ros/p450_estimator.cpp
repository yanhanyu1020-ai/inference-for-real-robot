#include <ros/ros.h>
#include "uav_estimator.h"

// Reuse the installed estimator and its validity checks. No UAV_controller is
// constructed, so this process owns no motion, mode, arming or parameter service.
int main(int argc, char** argv) {
    ros::init(argc, argv, "heterovla_p450_estimator");
    ros::NodeHandle nh("~");
    UAV_estimator estimator(nh);
    ros::spin();
    return 0;
}
