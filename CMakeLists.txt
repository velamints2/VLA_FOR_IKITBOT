cmake_minimum_required(VERSION 3.0.2)
project(obstacle_detection)

## 查找 catkin 宏和库
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
)

## Catkin 包配置
catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs cv_bridge
)

## 包含目录
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## 安装 Python 脚本
catkin_install_python(PROGRAMS
  src/ros_demo/obstacle_detector_node.py
  src/ros_demo/video_processor_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## 安装 launch 文件
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

## 安装配置文件
install(DIRECTORY configs/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/configs
  FILES_MATCHING PATTERN "*.yaml" PATTERN "*.txt"
)

## 安装模型文件 (可选，生产环境可能单独部署)
# install(DIRECTORY models/
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
#   FILES_MATCHING PATTERN "*.pt" PATTERN "*.onnx"
# )

