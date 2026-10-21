# GPS-Denied Localization and Anomaly Mapping with ROS

ROS 1 / MAVROS project combining LiDAR-based Hector SLAM, external-navigation odometry forwarding to ArduPilot, PatchCore anomaly detection, and RViz markers.

> **Safety:** This is a research/engineering example, not a flight-certified navigation system. Do not arm or fly a real vehicle based only on this repository. Validate estimator quality, coordinate conventions, sensor mounting, failsafes, and parameters in SITL, then conduct supervised ground tests with propellers removed.

## Features

- Hector SLAM publishes a local pose and map from a 2D LiDAR.
- `scripts/slam_to_fcu_odometry.py` converts Hector's `PoseStamped` pose to `nav_msgs/Odometry` and publishes to `/mavros/odometry/in`.
- PatchCore performs image anomaly detection and draws bounding boxes in a live camera window.
- Confirmed anomalies are associated with a recent SLAM pose and published as red RViz markers on `/anomaly_markers`.
- The existing cycle logic publishes green midpoint markers after the MAVROS local-height topic rises above and then falls below `HEIGHT_TRIGGER`.
- `launch/start.launch` starts MAVROS, Hector SLAM, the odometry bridge, anomaly detection, and RViz. Components can be disabled individually.

## Repository layout

```text
localization_and_anomaly_detection/
├── CMakeLists.txt
├── package.xml
├── launch/start.launch
├── scripts/
│   ├── slam_to_fcu_odometry.py
│   ├── anomaly_detection_node.py
│   └── train_patchcore.py
├── config/indoor_external_nav_minimal.param
├── docs/parameter_comparison.md
├── docs/parameter_diff.csv
├── models/README.md
├── dataset/train/normal/
├── dataset/test/anomaly/
├── rviz/anomaly_mapping.rviz
└── worlds/ and gazebo_models/ (simulation assets)
```

The trained checkpoint and raw videos are excluded from this GitHub-ready package. The checkpoint is too large for a normal GitHub repository; see [Model checkpoint](#model-checkpoint).

## Compatibility notes

- Intended ROS graph: ROS 1 Noetic / Ubuntu 20.04, MAVROS, Hector SLAM, ArduPilot Copter using EKF3.
- The anomaly script uses a newer `anomalib` PatchCore API. `anomalib 2.x` requires a newer Python runtime than the default Python 3.8 on Ubuntu 20.04/ROS Noetic. Do **not** assume the anomaly node will run in the stock Noetic interpreter. Use a compatible anomaly runtime or adapt the runtime boundary to your deployment. The SLAM-to-FCU bridge does not require anomalib.
- GPU acceleration is optional for inference, but recommended. PyTorch must match the host/Jetson CUDA stack.

## 1. Install ROS and system dependencies

Use a clean Ubuntu 20.04 + ROS Noetic environment for the ROS 1 graph. Install packages appropriate to your machine and LiDAR driver.

```bash
sudo apt update
sudo apt install -y \
  ros-noetic-desktop-full \
  ros-noetic-mavros ros-noetic-mavros-extras \
  ros-noetic-hector-mapping ros-noetic-rviz \
  ros-noetic-tf ros-noetic-cv-bridge \
  python3-catkin-tools python3-pip python3-rosdep \
  python3-opencv
```

Install GeographicLib datasets required by MAVROS:

```bash
sudo /opt/ros/noetic/lib/mavros/install_geographiclib_datasets.sh
```

Initialize rosdep if it has not been initialized before:

```bash
sudo rosdep init
rosdep update
```

Source ROS in each terminal:

```bash
source /opt/ros/noetic/setup.bash
```

Install the LiDAR driver that matches your hardware. Ensure it publishes `sensor_msgs/LaserScan`. This project cannot assume all LiDAR models use the same topic or frame ID.

## 2. Create the catkin workspace

Place the package directory inside a catkin workspace:

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
# Clone this repository here, or copy this package directory here.
# git clone https://github.com/<YOUR_GITHUB_USERNAME>/localization_and_anomaly_detection.git

cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source ~/catkin_ws/devel/setup.bash
```

To source it in every new terminal:

```bash
echo 'source /opt/ros/noetic/setup.bash' >> ~/.bashrc
echo 'source ~/catkin_ws/devel/setup.bash' >> ~/.bashrc
```

Make scripts executable:

```bash
chmod +x ~/catkin_ws/src/localization_and_anomaly_detection/scripts/*.py
```

## 3. Check ROS topics and TF before launch

Start the LiDAR and camera drivers first. Inspect actual topic names:

```bash
rostopic list
rostopic info /your_lidar_scan_topic
rostopic echo -n 1 /your_lidar_scan_topic
rostopic info /your_camera_topic
```

| Purpose | Default topic | Expected message |
|---|---|---|
| LiDAR scan | `/spur/laser/scan` | `sensor_msgs/LaserScan` |
| Hector SLAM pose | `/slam_out_pose` | `geometry_msgs/PoseStamped` |
| Camera | `/webcam/image_raw` | `sensor_msgs/Image` |
| MAVROS local height | `/mavros/local_position/pose` | `geometry_msgs/PoseStamped` |
| Anomaly markers | `/anomaly_markers` | `visualization_msgs/MarkerArray` |
| FCU external odometry | `/mavros/odometry/in` | `nav_msgs/Odometry` |

The LiDAR message's `header.frame_id` must match `laser_frame`; a valid transform from `base_frame` to the LiDAR frame must exist. Adjust the static transform in `start.launch` to match the measured sensor mounting.

## 4. Configure the PatchCore checkpoint

The repository does not include the approximately 281 MB trained checkpoint.

1. Install a PyTorch/CUDA build supported by your target computer.
2. Install an `anomalib` version compatible with the code and Python runtime.
3. Prepare normal training images in `dataset/train/normal/`.
4. Train:

   ```bash
   python3 scripts/train_patchcore.py
   ```

5. Copy the resulting checkpoint to `models/model.ckpt`, or pass its absolute path when launching:

   ```bash
   roslaunch localization_and_anomaly_detection start.launch \
     checkpoint:=/absolute/path/to/model.ckpt
   ```

The inference architecture is PatchCore (`wide_resnet50_2`, `layer2`, `layer3`). The checkpoint and inference architecture/anomalib version must match. Tune `SCORE_THRESHOLD` and `PIXEL_PERCENTILE` in `scripts/anomaly_detection_node.py` using held-out normal and anomalous examples; these thresholds are dataset-specific.

### Python runtime warning for anomalib

ROS Noetic on Ubuntu 20.04 normally uses Python 3.8, while anomalib 2.x uses Python 3.10 or newer. Installing anomalib 2.x into a separate Python 3.10 virtual environment does not automatically make it importable from the ROS Noetic process. The odometry bridge can run independently, but the anomaly node requires a compatible runtime strategy. Verify imports of `rospy`, `cv_bridge`, `torch`, and `anomalib` under the exact interpreter that starts the node before live operation.

## 5. Configure ArduPilot for GPS-denied ExternalNav

Do not overwrite the competition parameter file with the full `indoor.param` file. These files differ in RC endpoints, compass/IMU calibration, motor settings, flight modes, failsafes, serial ports, and other vehicle-specific values.

A reviewed subset from `indoor.param` is saved as `config/indoor_external_nav_minimal.param`. Confirm every parameter against your exact firmware and sensors before applying it.

| Parameter | Value in `indoor.param` | Purpose |
|---|---:|---|
| `GPS1_TYPE` | `0` | Disable GPS1 |
| `VISO_TYPE` | `1` | Enable MAVLink visual/external odometry support on supported firmware |
| `EK3_SRC1_POSXY` | `6` | ExternalNav horizontal position |
| `EK3_SRC1_VELXY` | `6` | ExternalNav horizontal velocity |
| `EK3_SRC1_POSZ` | `2` | Rangefinder altitude source |
| `EK3_SRC1_VELZ` | `0` | No vertical velocity source |
| `EK3_SRC1_YAW` | `6` | ExternalNav yaw |

These are not universal values. `EK3_SRC1_POSZ=2` assumes a suitable rangefinder is installed and configured. If altitude is supplied by the barometer instead, the appropriate source is typically `1`. Hector SLAM is planar and its z output must not be treated as trustworthy altitude.

Critical checks:

- Back up the current parameters before changing anything.
- The supplied `indoor.param` has `ARMING_CHECK=0`, which disables arming checks. Do not copy this blindly to a real aircraft.
- ExternalNav requires consistent coordinate frames, valid timestamps, sensible covariance, and an EKF origin. Set the EKF origin using the approved ground-station procedure for your vehicle/firmware.
- Reboot the flight controller after changing parameters when required by the firmware.
- The EKF source set must match the data being sent. This bridge estimates horizontal velocity from successive SLAM poses, but that estimate can drift or jump. Do not enable EKF fusion until the stream is verified.

Official references:
- [ArduPilot EKF source selection](https://ardupilot.org/copter/docs/common-ekf-sources.html)
- [ArduPilot non-GPS position estimation](https://ardupilot.org/dev/docs/mavlink-nongps-position-estimation.html)
- [ArduPilot Cartographer/ROS non-GPS navigation notes](https://ardupilot.org/dev/docs/ros-cartographer-slam.html)

## 6. Launch the system

Start the LiDAR and camera drivers, then launch the ROS graph:

```bash
roslaunch localization_and_anomaly_detection start.launch \
  fcu_url:=udp://127.0.0.1:14550@ \
  scan_topic:=/your_lidar_scan_topic \
  image_topic:=/your_camera_topic \
  laser_frame:=your_laser_frame \
  checkpoint:=/absolute/path/to/model.ckpt
```

For a serial-connected companion computer, use the correct MAVROS URL for the actual device (for example `/dev/ttyACM0:921600`, if that matches the hardware and MAVROS version). The exact URL syntax, port and baud must match the wiring.

Disable components during debugging:

```bash
# SLAM and RViz only: do not start MAVROS, odometry forwarding or inference.
roslaunch localization_and_anomaly_detection start.launch \
  start_mavros:=false start_odometry_bridge:=false \
  start_anomaly_detection:=false
```

For SITL, first configure ArduPilot SITL's MAVLink endpoint to match `fcu_url`. Do not assume the default UDP port matches every SITL setup.

## 7. Verify data flow before flight

```bash
rostopic type /slam_out_pose
rostopic hz /slam_out_pose
rostopic echo -n 1 /slam_out_pose

rostopic type /mavros/odometry/in
rostopic hz /mavros/odometry/in
rostopic echo -n 1 /mavros/odometry/in

rostopic echo -n 1 /mavros/state
rostopic echo -n 1 /anomaly_markers
```

The bridge warns if no SLAM pose has arrived or the stream is stale. Treat this as a navigation fault, not something to ignore.

## 8. How the odometry bridge works

1. Subscribes to Hector SLAM's `/slam_out_pose` (`PoseStamped`).
2. Copies position and orientation into `nav_msgs/Odometry` and assigns conservative configurable pose covariance values because `PoseStamped` has no covariance field.
3. Estimates translational velocity from consecutive poses and rotates horizontal velocity into the body frame.
4. Publishes to `/mavros/odometry/in`, MAVROS's odometry input topic for forwarding external odometry to the flight controller.
5. Does not invent altitude: Hector's 2D z value is not a replacement for a rangefinder or barometer.

Finite-difference velocity is sensitive to pose noise and SLAM jumps. Tune and validate it with recorded data before enabling EKF fusion. For production use, prefer a tested estimator that provides timestamped pose, velocity, and realistic covariance.

## 9. Anomaly markers

- The inference node subscribes to the camera and most recent SLAM pose.
- A location is recorded after `REQUIRED_CONSECUTIVE_FRAMES` consecutive detections.
- Red markers represent confirmed anomaly locations.
- The existing cycle logic adds green midpoint markers after the MAVROS local-height topic rises above and then falls below `HEIGHT_TRIGGER`.
- RViz uses fixed frame `map` and displays `/anomaly_markers`.

If markers do not appear, verify the SLAM pose topic (`geometry_msgs/PoseStamped`) and the image topic, and that RViz's fixed frame matches `MAP_FRAME`.

## 10. Parameter comparison

See [`docs/parameter_comparison.md`](docs/parameter_comparison.md) for the GPS-denied navigation-focused list of changed/missing parameters and [`docs/parameter_diff.csv`](docs/parameter_diff.csv) for spreadsheet-friendly data.

The supplied files also differ in vehicle-specific calibration, RC settings, flight modes, motor configuration, and failsafes. Transfer only values intended and validated for the same vehicle.

## 11. Publish to GitHub

1. Create an empty repository on GitHub (do not initialize it with a README if you plan to push this folder as-is).
2. Extract this ZIP and enter the extracted `localization_and_anomaly_detection` directory.
3. Review `git status` and confirm no private logs, videos, or model checkpoints are staged.
4. Run:

   ```bash
   git init
   git add .
   git status
   git commit -m "Prepare ROS GPS-denied localization and anomaly mapping"
   git branch -M main
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY>.git
   git push -u origin main
   ```

Replace the remote URL with the repository URL shown by GitHub. The `.gitignore` excludes model checkpoints, raw recordings, build outputs, and Python cache files. Dataset images currently included in this package will be uploaded unless you remove them or add their directory to `.gitignore`.

## License

MIT. Review third-party model, dataset, and simulation-asset licenses before redistribution.
