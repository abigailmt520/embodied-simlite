#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ros_bridge.py
=============
Embodied-SimLite ROS 2 bridge layer (refactored from the old sim_ros2_bridgeV1.0.py).

Responsibility boundary under the first principle (full decoupling):
    - physics rollout and AI decision are fully consolidated into inference_server.py (PPO + embodied_env.py);
    - this bridge only does "protocol translation": translating the twin-state broadcast into ROS 2 topics, and translating ROS 2 control into
      a manual-override command the inference gateway recognizes.

Data flow:
    Downstream (subscribe): the inference gateway's /ws broadcasts the new nested contract of get_render_state()
                  → publish /odom, /scan, and broadcast tf (odom→base_footprint→base_link/laser_frame).
    Upstream (publish): subscribe to /cmd_vel(_nav) → send {"cmd_vel":{...}} over /ws to trigger the 2s manual override.

Contract alignment (key change):
    the old version read the flat fields data["ox"]/data["oyaw"]/data["lidar"];
    the new version reads nested fields: odometry from data["odom"] (with real slip drift, consistent with the platform's "true-fork odometry"),
    LiDAR from data["lidar"] (real ranging m).
    ⚠ /odom must publish the "odometry" data["odom"] rather than the "truth" data["robot"] — publishing truth by mistake
    would revive the "odometry≡truth" fake instrument at the ROS layer, contradicting the platform's true-fork odometry.
"""

import os
import json
import math
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
import websocket
from geometry_msgs.msg import Twist, TransformStamped, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster

# inference-gateway WebSocket address (unified endpoint /ws, replaces the old /ws/simulation)
GATEWAY_WS_URL = os.environ.get("SIM_GATEWAY_WS", "ws://127.0.0.1:8000/ws")


def get_quaternion_from_euler(yaw):
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


class EmbodiedRos2Bridge(Node):
    def __init__(self):
        super().__init__('embodied_sim_bridge')

        # —— control downstream: subscribe to Nav2 / teleop velocity commands ——
        self.create_subscription(TwistStamped, '/cmd_vel_nav', self.cmd_stamped_callback, 10)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_twist_callback, 10)

        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10
        )
        self.scan_pub = self.create_publisher(LaserScan, '/scan', sensor_qos)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.robot_base_frame = "base_footprint"

        self.ws = None
        self.log_counter = 0
        self.last_step = -1   # dedup by render_state's step, avoiding republishing the same frame

        self.connect_websocket()

    # ------------------------------------------------------------------
    # WebSocket connection management
    # ------------------------------------------------------------------
    def connect_websocket(self):
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(GATEWAY_WS_URL,
                                         on_open=self.on_open,
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close)
        wst = threading.Thread(target=self.ws_thread)
        wst.daemon = True
        wst.start()
        self.get_logger().info(f"🔗 ROS 2 bridge starting, connecting to the inference gateway {GATEWAY_WS_URL} ...")

    def ws_thread(self):
        while True:
            self.ws.run_forever()
            self.get_logger().error("❌ not connected to the inference gateway, retrying in 3s...")
            time.sleep(3)

    def on_open(self, ws):
        self.get_logger().info("✅ successfully connected to the inference gateway!")
        self.last_step = -1

    def on_close(self, ws, close_status_code, close_msg):
        self.get_logger().warn("⚠️ WebSocket connection closed!")

    # ------------------------------------------------------------------
    # control downstream: /cmd_vel → manual-override command (the inference-gateway side converts to normalized action and preempts RL)
    # ------------------------------------------------------------------
    def cmd_twist_callback(self, msg):
        self.process_cmd(msg.linear.x, msg.angular.z)

    def cmd_stamped_callback(self, msg):
        self.process_cmd(msg.twist.linear.x, msg.twist.angular.z)

    def process_cmd(self, v_x, w_z):
        if abs(v_x) > 0.01 or abs(w_z) > 0.01:
            self.log_counter += 1
            if self.log_counter % 5 == 0:
                self.get_logger().info(
                    f"🕹️ [manual override sent] linear: {v_x:.2f} m/s, angular: {w_z:.2f} rad/s")
        try:
            # pass through real physical quantities; normalization and the 2s override window are handled by the gateway's OverrideController
            self.ws.send(json.dumps({"cmd_vel": {"linear": float(v_x), "angular": float(w_z)}}))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # state upstream: parse the new nested contract → publish /odom, /scan, tf
    # ------------------------------------------------------------------
    def on_message(self, ws, message):
        data = json.loads(message)
        current_time = self.get_clock().now().to_msg()

        # —— odometry + tf: read the new data["odom"] (with real slip drift; fall back to robot when the old contract has no odom) ——
        odom_src = data.get("odom") or data.get("robot")
        if odom_src is not None:
            ox = float(odom_src["x"])    # odometry (with drift), not truth — consistent with the platform's true-fork odometry
            oy = float(odom_src["y"])
            oyaw = float(odom_src["theta"])

            odom = Odometry()
            odom.header.stamp = current_time
            odom.header.frame_id = "odom"
            odom.child_frame_id = self.robot_base_frame
            odom.pose.pose.position.x = ox
            odom.pose.pose.position.y = oy
            q = get_quaternion_from_euler(oyaw)
            odom.pose.pose.orientation.x = q[0]
            odom.pose.pose.orientation.y = q[1]
            odom.pose.pose.orientation.z = q[2]
            odom.pose.pose.orientation.w = q[3]
            self.odom_pub.publish(odom)

            t0 = TransformStamped()
            t0.header.stamp = current_time
            t0.header.frame_id = "odom"
            t0.child_frame_id = self.robot_base_frame
            t0.transform.translation.x = ox
            t0.transform.translation.y = oy
            t0.transform.translation.z = 0.15
            t0.transform.rotation.x = q[0]
            t0.transform.rotation.y = q[1]
            t0.transform.rotation.z = q[2]
            t0.transform.rotation.w = q[3]

            t1 = TransformStamped()
            t1.header.stamp = current_time
            t1.header.frame_id = self.robot_base_frame
            t1.child_frame_id = "base_link"
            t1.transform.rotation.w = 1.0

            t2 = TransformStamped()
            t2.header.stamp = current_time
            t2.header.frame_id = self.robot_base_frame
            t2.child_frame_id = "laser_frame"
            t2.transform.translation.z = 0.5
            t2.transform.rotation.w = 1.0

            self.tf_broadcaster.sendTransform([t0, t1, t2])

        # —— LiDAR: read the new data["lidar"] (real ranging m) + data["lidar_range"] ——
        lidar = data.get("lidar")
        step = data.get("step", None)
        if lidar and step != self.last_step:
            self.last_step = step
            n = len(lidar)

            scan = LaserScan()
            scan.header.stamp = current_time
            scan.header.frame_id = "laser_frame"
            # env ray offset linspace(-π, π, N, endpoint=False): angle_min=-π, increment=2π/N
            scan.angle_min = -math.pi
            scan.angle_increment = (2.0 * math.pi) / n
            scan.angle_max = scan.angle_min + scan.angle_increment * (n - 1)
            scan.range_min = 0.0
            scan.range_max = float(data.get("lidar_range", 5.0))
            scan.ranges = [float(r) for r in lidar]

            self.scan_pub.publish(scan)

    def on_error(self, ws, error):
        pass


def main(args=None):
    rclpy.init(args=args)
    bridge = EmbodiedRos2Bridge()
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
