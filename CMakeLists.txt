cmake_minimum_required(VERSION 3.0.2)
project(localization_and_anomaly_detection)

find_package(catkin REQUIRED COMPONENTS
  rospy geometry_msgs nav_msgs sensor_msgs visualization_msgs tf
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/slam_to_fcu_odometry.py
  scripts/anomaly_detection_node.py
  scripts/train_patchcore.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch rviz worlds gazebo_models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  OPTIONAL
)
