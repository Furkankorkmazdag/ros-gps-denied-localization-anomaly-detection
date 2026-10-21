#!/usr/bin/env python3
"""Forward Hector SLAM pose estimates to ArduPilot through MAVROS ODOMETRY.

Input:  geometry_msgs/PoseStamped, normally /slam_out_pose.
Output: nav_msgs/Odometry to /mavros/odometry/in.

Hector SLAM is planar; its z position is not a reliable altitude estimate.
Configure ArduPilot to use an independent rangefinder/barometer for POSZ.
Test in SITL before any real flight.
"""
import math
import time

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion


class SlamToFcuOdometry:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/slam_out_pose")
        self.output_topic = rospy.get_param("~output_topic", "/mavros/odometry/in")
        self.parent_frame = rospy.get_param("~parent_frame", "map")
        self.child_frame = rospy.get_param("~child_frame", "base_link")
        self.stale_timeout = float(rospy.get_param("~stale_timeout", 0.5))
        self.max_velocity = float(rospy.get_param("~max_velocity", 5.0))
        self.pose = None
        self.last_wall_time = None
        self.last_xyz = None
        self.last_yaw = None
        self.pub = rospy.Publisher(self.output_topic, Odometry, queue_size=10)
        self.sub = rospy.Subscriber(
            self.input_topic, PoseStamped, self.pose_callback,
            queue_size=5
        )
        self.timer = rospy.Timer(rospy.Duration(0.1), self.watchdog)
        rospy.loginfo("SLAM -> FCU odometry bridge: %s -> %s",
                      self.input_topic, self.output_topic)

    def pose_callback(self, msg):
        now_wall = time.monotonic()
        pose = msg.pose
        xyz = (pose.position.x, pose.position.y, pose.position.z)
        q = pose.orientation
        yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))[2]

        odom = Odometry()
        odom.header.stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()
        odom.header.frame_id = msg.header.frame_id if msg.header.frame_id else self.parent_frame
        odom.child_frame_id = self.child_frame
        odom.pose.pose = pose
        # PoseStamped has no covariance. Use conservative configurable diagonal
        # variances instead of incorrectly implying zero uncertainty.
        pose_var_xy = float(rospy.get_param("~pose_variance_xy", 0.10))
        pose_var_z = float(rospy.get_param("~pose_variance_z", 1.00))
        pose_var_roll_pitch = float(rospy.get_param("~pose_variance_roll_pitch", 0.25))
        pose_var_yaw = float(rospy.get_param("~pose_variance_yaw", 0.20))
        odom.pose.covariance[0] = pose_var_xy
        odom.pose.covariance[7] = pose_var_xy
        odom.pose.covariance[14] = pose_var_z
        odom.pose.covariance[21] = pose_var_roll_pitch
        odom.pose.covariance[28] = pose_var_roll_pitch
        odom.pose.covariance[35] = pose_var_yaw

        if self.last_xyz is not None and self.last_wall_time is not None:
            dt = now_wall - self.last_wall_time
            if 0.005 <= dt <= 1.0:
                vx_world = (xyz[0] - self.last_xyz[0]) / dt
                vy_world = (xyz[1] - self.last_xyz[1]) / dt
                vz_world = (xyz[2] - self.last_xyz[2]) / dt
                # nav_msgs/Odometry twist is expressed in child_frame_id.
                c, s = math.cos(yaw), math.sin(yaw)
                vx_body = c * vx_world + s * vy_world
                vy_body = -s * vx_world + c * vy_world
                speed = math.sqrt(vx_body * vx_body + vy_body * vy_body + vz_world * vz_world)
                if speed > self.max_velocity and speed > 0:
                    scale = self.max_velocity / speed
                    vx_body *= scale
                    vy_body *= scale
                    vz_world *= scale
                odom.twist.twist.linear.x = vx_body
                odom.twist.twist.linear.y = vy_body
                odom.twist.twist.linear.z = vz_world
                # Conservative default velocity covariance.
                odom.twist.covariance[0] = 0.25
                odom.twist.covariance[7] = 0.25
                odom.twist.covariance[14] = 0.5
                odom.twist.covariance[21] = 0.1
                odom.twist.covariance[28] = 0.1
                odom.twist.covariance[35] = 0.2

        self.pub.publish(odom)
        self.pose = odom
        self.last_xyz = xyz
        self.last_yaw = yaw
        self.last_wall_time = now_wall

    def watchdog(self, _event):
        if self.last_wall_time is None:
            rospy.logwarn_throttle(5.0, "Waiting for SLAM pose on %s", self.input_topic)
            return
        age = time.monotonic() - self.last_wall_time
        if age > self.stale_timeout:
            rospy.logerr_throttle(
                1.0, "SLAM pose is stale (%.2f s); FCU odometry is not being refreshed.", age
            )


if __name__ == "__main__":
    rospy.init_node("slam_to_fcu_odometry")
    SlamToFcuOdometry()
    rospy.spin()
