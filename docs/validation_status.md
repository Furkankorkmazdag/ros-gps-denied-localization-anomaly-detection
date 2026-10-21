# Validation status

Static checks performed when preparing this package:
- Python syntax compilation for the ROS scripts.
- XML parsing for `package.xml` and `launch/start.launch`.
- Parameter comparison generated from the two supplied `.param` files.

Not performed in this environment:
- Catkin build against a live ROS Noetic installation.
- Runtime verification of the installed MAVROS odometry plugin and FCU firmware.
- SITL flight tests, sensor TF validation, camera inference, or real-flight testing.

Run the launch and topic checks in README.md on the target system. Do not treat static checks as flight validation.
