#!/usr/bin/env python

#
# Copyright (c) 2024 Numurus, LLC <https://www.numurus.com>.
#
# This file is part of nepi-engine
# (see https://github.com/nepi-engine).
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
#

# NEPI IDX Node for Zed cameras
# TODO: Finish converting this to a real IDX node leveraging idx_sensor_if.py etc.

###################################################
# NEPI NavPose Axis Info
# x+ axis is forward
# y+ axis is right
# z+ axis is down
# roll: RHR about x axis
# pitch: RHR about y axis
# yaw: RHR about z axis
#####################################################

### Set the namespace before importing rospy
import os
os.environ["ROS_NAMESPACE"] = "/nepi/s2x"

import time
import subprocess
import os
import rospy
import dynamic_reconfigure.client
import numpy as np
import cv2
import math
import tf

from datetime import datetime
from std_msgs.msg import UInt8, Empty, String, Bool, Float32
from sensor_msgs.msg import Image, PointCloud2
from nepi_ros_interfaces.msg import IDXStatus, RangeWindow, SaveDataStatus, SaveData, SaveDataRate
from nepi_ros_interfaces.srv import IDXCapabilitiesQuery, IDXCapabilitiesQueryResponse
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, PoseStamped, QuaternionStamped
from dynamic_reconfigure.msg import Config
from rospy.numpy_msg import numpy_msg
from cv_bridge import CvBridge

#########################################
# ROS NAMESPACE SETUP
#########################################

NEPI_BASE_NAMESPACE = "/nepi/s2x/"

  #######################
  # Process Functions

### Function to Convert Quaternion Attitude to Roll, Pitch, Yaw Degrees
def convert_quat2rpy(xyzw_attitude):
  rpy_attitude_rad = tf.transformations.euler_from_quaternion(xyzw_attitude)
  rpy_attitude_ned_deg = np.array(rpy_attitude_rad) * 180/math.pi
  roll_deg = rpy_attitude_ned_deg[0] 
  pitch_deg = rpy_attitude_ned_deg[1] 
  yaw_deg = rpy_attitude_ned_deg[2]
  return rpy_attitude_ned_deg

### Function to Convert Roll, Pitch, Yaw Degrees to Quaternion Attitude
def convert_rpy2quat(rpy_attitude_ned_deg):
  roll_deg = rpy_attitude_ned_deg[0] 
  pitch_deg = rpy_attitude_ned_deg[1] 
  yaw_deg = rpy_attitude_ned_deg[2]
  xyzw_attitude = tf.transformations.quaternion_from_euler(math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg))
  return xyzw_attitude

### Function to find a topic
def find_topic(topic_name):
  topic = ""
  topic_list=rospy.get_published_topics(namespace='/')
  for topic_entry in topic_list:
    if topic_entry[0].find(topic_name) != -1:
      topic = topic_entry[0]
  return topic

### Function to check for a topic 
def check_for_topic(topic_name):
  topic_exists = True
  topic=find_topic(topic_name)
  if topic == "":
    topic_exists = False
  return topic_exists

### Function to wait for a topic
def wait_for_topic(topic_name):
  topic = ""
  while topic == "" and not rospy.is_shutdown():
    topic=find_topic(topic_name)
    time.sleep(.1)
  return topic

class ZedCameraNode(object) :
  #########################################
  # DRIVER SETTINGS
  #########################################

  #Define Sensor Native Parameters
  SENSOR_RES_OPTION_LIST = [0,1,2,3]  # Maps to IDX Res Options 0-3
  SENSOR_MAX_BRIGHTNESS = 8
  SENSOR_MAX_CONTRAST = 8
  SENSOR_MAX_FRAMERATE_FPS = 30
  SENSOR_MIN_THRESHOLD = 1
  SENSOR_MAX_THRESHOLD = 100
  DEFAULT_SENSOR_MIN_RANGE_M = 0
  DEFAULT_SENSOR_MAX_RANGE_M = 20

  #Set Initialize IDX Parameters
  IDX_RES_MODE = 1
  IDX_FRAMERATE_MODE = 3
  IDX_BRIGHTNESS_RATIO = 0.5
  IDX_CONTRAST_RATIO = 0.5
  IDX_THRESHOLD_RATIO = 0.5
  IDX_MIN_RANGE_RATIO=0.02 
  IDX_MAX_RANGE_RATIO=0.15 

  def __init__(self):
    # This parameter should be automatically set by idx_sensor_mgr
    zed_type = rospy.get_param('~zed_type', 'zed2')

    self.node_name = rospy.get_name().split('/')[-1]

    # Now assign values for the Zed ROS Wrapper topics
    ZED_BASE_NAMESPACE = rospy.get_namespace() + zed_type + "/zed_node/"
    # Zed control topics
    # ZED_PARAMETER_UPDATES_TOPIC = ZED_BASE_NAMESPACE + "parameter_updates"
    # Zed data stream topics
    ZED_COLOR_2D_IMAGE_TOPIC = ZED_BASE_NAMESPACE + "left/image_rect_color"
    ZED_BW_2D_IMAGE_TOPIC = ZED_BASE_NAMESPACE + "left/image_rect_gray"
    ZED_DEPTH_MAP_TOPIC = ZED_BASE_NAMESPACE + "depth/depth_registered"
    ZED_POINTCLOUD_TOPIC = ZED_BASE_NAMESPACE + "point_cloud/cloud_registered"
    ZED_ODOM_TOPIC = ZED_BASE_NAMESPACE + "odom"

    ZED_MIN_RANGE_PARAM = ZED_BASE_NAMESPACE + "depth/min_depth"
    ZED_MAX_RANGE_PARAM = ZED_BASE_NAMESPACE + "depth/max_depth"

    # And IDX topics
    NEPI_IDX_SENSOR_NAME = zed_type + "_stereo_camera"
    NEPI_IDX_SENSOR_NAMESPACE = NEPI_BASE_NAMESPACE + NEPI_IDX_SENSOR_NAME
    NEPI_IDX_NAMESPACE = NEPI_IDX_SENSOR_NAMESPACE + "/idx/"

    ### NEPI IDX NavPose Publish Topic
    NEPI_IDX_NAVPOSE_ODOM_TOPIC = NEPI_IDX_NAMESPACE + "odom"
    # NEPI IDX capabilities query service
    NEPI_IDX_CAPABILITY_REPORT_SERVICE = NEPI_IDX_NAMESPACE + "capabilities_query"
    NEPI_IDX_CAPABILITY_NAVPOSE_TOPIC = NEPI_IDX_NAMESPACE + "navpose_support"
    self.NEPI_IDX_CAPABILITY_NAVPOSE = 2 # Bit Mask [GPS,ODOM,HEADING]
    # NEPI IDX status and control topics
    NEPI_IDX_STATUS_TOPIC = NEPI_IDX_NAMESPACE + "status"
    NEPI_IDX_SET_BRIGHTNESS_TOPIC = NEPI_IDX_NAMESPACE + "set_brightness"
    NEPI_IDX_SET_CONTRAST_TOPIC = NEPI_IDX_NAMESPACE + "set_contrast"
    #NEPI_IDX_SET_FRAMERATE_MODE_TOPIC = NEPI_IDX_NAMESPACE + "set_framerate_mode"
    #NEPI_IDX_SET_RESOLUTION_MODE_TOPIC = NEPI_IDX_NAMESPACE + "set_resolution_mode"
    NEPI_IDX_SET_THRESHOLDING_TOPIC = NEPI_IDX_NAMESPACE + "set_thresholding"
    NEPI_IDX_SET_RANGE_WINDOW_TOPIC = NEPI_IDX_NAMESPACE + "set_range_window"
    # NEPI IDX data stream topics
    NEPI_IDX_COLOR_2D_IMAGE_TOPIC = NEPI_IDX_NAMESPACE + "color_2d_image"
    NEPI_IDX_BW_2D_IMAGE_TOPIC = NEPI_IDX_NAMESPACE + "bw_2d_image"
    NEPI_IDX_DEPTH_MAP_TOPIC = NEPI_IDX_NAMESPACE + "depth_map"
    NEPI_IDX_DEPTH_IMAGE_TOPIC = NEPI_IDX_NAMESPACE + "depth_image"
    NEPI_IDX_POINTCLOUD_TOPIC = NEPI_IDX_NAMESPACE + "pointcloud"
    NEPI_IDX_POINTCLOUD_IMAGE_TOPIC = NEPI_IDX_NAMESPACE + "pointcloud_image"
    # NEPI IDX save data subscriber topics
    self.SAVE_FOLDER = "/mnt/nepi_storage/data/"
    NEPI_IDX_SAVE_DATA_TOPIC = NEPI_BASE_NAMESPACE + "save_data"
    NEPI_IDX_SAVE_DATA_PREFIX_TOPIC = NEPI_BASE_NAMESPACE + "save_data_prefix"
    NEPI_IDX_SAVE_DATA_RATE_TOPIC = NEPI_BASE_NAMESPACE + "save_data_rate"

    self.idx_capability_navpose_pub = rospy.Publisher(NEPI_IDX_CAPABILITY_NAVPOSE_TOPIC, UInt8, queue_size=1)
    self.idx_navpose_odom_pub = rospy.Publisher(NEPI_IDX_NAVPOSE_ODOM_TOPIC, Odometry, queue_size=1)
    self.idx_status_pub = rospy.Publisher(NEPI_IDX_STATUS_TOPIC, IDXStatus, queue_size=1, latch=True)
    self.idx_color_2d_image_pub = rospy.Publisher(NEPI_IDX_COLOR_2D_IMAGE_TOPIC, Image, queue_size=1)
    self.idx_bw_2d_image_pub = rospy.Publisher(NEPI_IDX_BW_2D_IMAGE_TOPIC, Image, queue_size=1)
    self.idx_depth_map_pub = rospy.Publisher(NEPI_IDX_DEPTH_MAP_TOPIC, Image, queue_size=1)
    self.idx_depth_image_pub = rospy.Publisher(NEPI_IDX_DEPTH_IMAGE_TOPIC, Image, queue_size=1)
    self.idx_pointcloud_pub = rospy.Publisher(NEPI_IDX_POINTCLOUD_TOPIC, PointCloud2, queue_size=1)
    self.idx_pointcloud_image_pub = rospy.Publisher(NEPI_IDX_POINTCLOUD_IMAGE_TOPIC, Image, queue_size=1)

    self.idx_status_msg=IDXStatus()
    self.idx_capabilities_report = IDXCapabilitiesQueryResponse()
    self.idx_save_data = False
    self.idx_save_data_prefix = ""
    self.idx_save_data_rate = 1.0
    self.idx_capability_pub_interval = 1
    self.idx_save_data_status_pub_interval = 1
    self.save_data_timer = 0.1

    self.color_2d_image_msg = None
    self.bw_2d_image_msg = None
    self.depth_map_msg = None
    self.depth_image_msg = None
    self.pointcloud_msg = None

    rospy.loginfo("Starting Initialization")

    # Run the correct zed_ros_wrapper launch file
    zed_launchfile = zed_type + '.launch'
    zed_ros_wrapper_run_cmd = ['roslaunch', 'zed_wrapper', zed_launchfile]
    # TODO: Some process management for the Zed ROS wrapper
    self.zed_ros_wrapper_proc = subprocess.Popen(zed_ros_wrapper_run_cmd)

    # Now that Zed SDK is started, we can set up the reconfig client
    self.zed_dynamic_reconfig_client = dynamic_reconfigure.client.Client(ZED_BASE_NAMESPACE, timeout=30)

    # Wait for zed odom topic (indicates Zed ROS Wrapper is running)
    ##############################
    rospy.loginfo("Waiting for ZED odom message to publish on " + ZED_ODOM_TOPIC)
    # Publish IDX NavPose supported topics
    wait_for_topic(ZED_ODOM_TOPIC)
    rospy.Subscriber(ZED_ODOM_TOPIC, Odometry, self.idx_odom_topic_callback)
    # Wait for zed depth topic
    rospy.loginfo("Waiting for topic: " + ZED_DEPTH_MAP_TOPIC)
    wait_for_topic(ZED_DEPTH_MAP_TOPIC)
    # Initialize IDX status msg and sensor
    self.idx_status_msg.idx_controls = True
    self.idx_status_msg.auto = False
    self.idx_status_msg.resolution_mode = self.IDX_RES_MODE  # Not sure if this is adjustable
    self.idx_status_msg.framerate_mode = self.IDX_FRAMERATE_MODE # Not sure if this is adjustable
    self.idx_status_msg.brightness = self.IDX_BRIGHTNESS_RATIO
    self.update_sensor_brightness(self.idx_status_msg.brightness)
    self.idx_status_msg.contrast = self.IDX_CONTRAST_RATIO
    self.update_sensor_contrast(self.idx_status_msg.contrast)  
    self.idx_status_msg.thresholding = self.IDX_THRESHOLD_RATIO
    self.update_sensor_thresholding(self.IDX_THRESHOLD_RATIO)
    self.idx_status_msg.range_window.start_range = self.IDX_MIN_RANGE_RATIO 
    self.idx_status_msg.range_window.stop_range = self.IDX_MAX_RANGE_RATIO
    self.idx_status_msg.min_range_m = rospy.get_param(ZED_MIN_RANGE_PARAM, self.DEFAULT_SENSOR_MIN_RANGE_M)
    self.idx_status_msg.max_range_m = rospy.get_param(ZED_MAX_RANGE_PARAM, self.DEFAULT_SENSOR_MAX_RANGE_M)
    self.idx_status_msg.frame_3d = "nepi_center_frame"
    self.idx_status_pub_callback()

    # Start IDX Subscribers
    rospy.Subscriber(NEPI_IDX_SET_BRIGHTNESS_TOPIC, Float32, self.idx_set_brightness_callback)
    rospy.Subscriber(NEPI_IDX_SET_CONTRAST_TOPIC, Float32, self.idx_set_contrast_callback)
    #rospy.Subscriber(NEPI_IDX_SET_FRAMERATE_MODE_TOPIC, UInt8, idx_set_framerate_mode_callback)
    #rospy.Subscriber(NEPI_IDX_SET_RESOLUTION_MODE_TOPIC, UInt8, idx_set_resolution_mode_callback)
    rospy.Subscriber(NEPI_IDX_SET_THRESHOLDING_TOPIC, Float32, self.idx_set_thresholding_callback)
    rospy.Subscriber(NEPI_IDX_SET_RANGE_WINDOW_TOPIC, RangeWindow, self.idx_set_range_window_callback)

    rospy.Subscriber(NEPI_IDX_SAVE_DATA_TOPIC, SaveData, self.idx_save_data_callback)
    rospy.Subscriber(NEPI_IDX_SAVE_DATA_PREFIX_TOPIC, String, self.idx_save_data_prefix_callback)
    rospy.Subscriber(NEPI_IDX_SAVE_DATA_RATE_TOPIC, SaveDataRate, self.idx_save_data_rate_callback)
    
    # Populate and advertise IDX Capability Report
    self.idx_capabilities_report.has_auto_adjustment
    self.idx_capabilities_report.adjustable_resolution = False # Pending callback implementation
    self.idx_capabilities_report.adjustable_framerate = False # Pending callback implementation
    self.idx_capabilities_report.adjustable_contrast = True
    self.idx_capabilities_report.adjustable_brightness = True
    self.idx_capabilities_report.adjustable_thresholding = True
    self.idx_capabilities_report.adjustable_range = True
    self.idx_capabilities_report.has_color_2d_image = True
    self.idx_capabilities_report.has_bw_2d_image = True
    self.idx_capabilities_report.has_depth_map = True
    self.idx_capabilities_report.has_depth_image = True 
    self.idx_capabilities_report.has_pointcloud_image = False # TODO: Create this data
    self.idx_capabilities_report.has_pointcloud = True
    rospy.Service(NEPI_IDX_CAPABILITY_REPORT_SERVICE, IDXCapabilitiesQuery, self.idx_capabilities_query_callback)
    rospy.Timer(rospy.Duration(self.idx_capability_pub_interval), self.idx_capability_pub_callback)
    rospy.Timer(rospy.Duration(0.1), self.idx_save_data_pub_callback)
    rospy.loginfo("Starting Zed IDX subscribers and publishers")
    rospy.Subscriber(ZED_COLOR_2D_IMAGE_TOPIC, Image, self.color_2d_image_callback, queue_size = 1)
    rospy.Subscriber(ZED_BW_2D_IMAGE_TOPIC, Image, self.bw_2d_image_callback, queue_size = 1)
    rospy.Subscriber(ZED_DEPTH_MAP_TOPIC, numpy_msg(Image), self.depth_map_callback, queue_size = 1)
    rospy.Subscriber(ZED_POINTCLOUD_TOPIC, PointCloud2, self.pointcloud_callback, queue_size = 1)
    rospy.loginfo("Initialization Complete")

    rospy.spin()

  ##############################
  # IDX Capabilities Topic Publishers
  ### Callback to publish IDX capabilities lists
  def idx_capability_pub_callback(self, _):
    if not rospy.is_shutdown():
      self.idx_capability_navpose_pub.publish(data=self.NEPI_IDX_CAPABILITY_NAVPOSE)

  ##############################
  # IDX Data Saver
  ### Callback to save data at set rate
  ### This is just a quick fix to add some save functionality.  Will save save same data if not updated
  def idx_save_data_pub_callback(self, _):
    save_data_interval = 1/self.idx_save_data_rate
    if self.save_data_timer < save_data_interval:
      self.save_data_timer = self.save_data_timer + 0.1
    else:
      if self.idx_save_data is True:
        date_str=datetime.utcnow().strftime('%Y-%m-%d')
        time_str=datetime.utcnow().strftime('%H%M%S')
        ms_str =datetime.utcnow().strftime('%f')[:-3]
        dt_str = (date_str + "T" + time_str + "." + ms_str)
        if self.color_2d_image_msg is not None:
          #Convert image from ros to cv2
          bridge = CvBridge()
          cv_image = bridge.imgmsg_to_cv2(self.color_2d_image_msg, "bgr8")
          # Saving image to file type
          image_filename=self.SAVE_FOLDER + self.idx_save_data_prefix + dt_str + '_' + self.node_name + '_2d_color_image.png'
          rospy.logdebug("Saving image to file")
          cv2.imwrite(image_filename,cv_image)
        if self.depth_image_msg is not None:
          #Convert image from ros to cv2
          bridge = CvBridge()
          cv_image = bridge.imgmsg_to_cv2(self.depth_image_msg, "bgr8")
          # Saving image to file type
          image_filename=self.SAVE_FOLDER + self.idx_save_data_prefix + dt_str + '_' + self.node_name + '_depth_image.png'
          rospy.logdebug("Saving image to file")
          cv2.imwrite(image_filename,cv_image)
      save_data_timer = 0
    #print(save_data_timer)

  ### Callback to publish idx odom topic
  def idx_odom_topic_callback(self, odom_msg):
    # TODO: Need to convert data from zed odom ref frame to nepi ref frame
    if not rospy.is_shutdown():
      self.idx_navpose_odom_pub.publish(odom_msg)

  #######################
  # Driver Status Publishers Functions

  ### function to publish IDX status message and updates
  def idx_status_pub_callback(self):
    if not rospy.is_shutdown():
      self.idx_status_pub.publish(self.idx_status_msg)

  #######################
  # Driver Save Data Subscribers

  ### callback to update save data setting
  def idx_save_data_callback(self, save_data_msg):
    self.idx_save_data = save_data_msg.save_continuous
    rospy.loginfo("Updating save data to: " + str(self.idx_save_data))

  ### callback to update save data prefix setting
  def idx_save_data_prefix_callback(self, save_data_prefix_msg):
    self.idx_save_data_prefix = save_data_prefix_msg.data
    rospy.loginfo("Updating save data prefix to: " + self.idx_save_data_prefix)

  ### callback to update save data rate setting
  def idx_save_data_rate_callback(self, save_data_rate_msg):
    self.idx_save_data_rate = save_data_rate_msg.save_rate_hz
    rospy.loginfo("Updating save data rate to: " + str(self.idx_save_data_rate))
  
  #######################
  # Driver Control Subscribers Functions

  ### callback to get and apply brightness control
  def idx_set_brightness_callback(self, brightness_msg):
    rospy.loginfo(brightness_msg)
    idx_brightness_ratio = brightness_msg.data
    # udpate sensor native values
    self.update_sensor_brightness(idx_brightness_ratio)
    # publish IDX status update
    self.idx_status_msg.brightness = idx_brightness_ratio
    self.idx_status_pub_callback()

  def update_sensor_brightness(self, brightness_ratio):
    # Sensor Specific
    sensor_brightness_val = int(float(self.SENSOR_MAX_BRIGHTNESS)*brightness_ratio)
    self.zed_dynamic_reconfig_client.update_configuration({"brightness":sensor_brightness_val})
    
  ### callback to get and apply contrast control
  def idx_set_contrast_callback(self, contrast_msg):
    rospy.loginfo(contrast_msg)
    idx_contrast_ratio = contrast_msg.data
    # udpate sensor native values
    self.update_sensor_contrast(idx_contrast_ratio)
    # publish IDX status update
    self.idx_status_msg.contrast = idx_contrast_ratio
    self.idx_status_pub_callback()

  def update_sensor_contrast(self, contrast_ratio):
    # Sensor Specific
    sensor_contrast_val = int(float(self.SENSOR_MAX_CONTRAST)*contrast_ratio)
    self.zed_dynamic_reconfig_client.update_configuration({"contrast":sensor_contrast_val})
    
  def idx_set_thresholding_callback(self, thresholding_msg):
    idx_thresholding_ratio = thresholding_msg.data
    # udpate sensor native values
    self.update_sensor_thresholding(idx_thresholding_ratio)
    # publish IDX status update
    self.idx_status_msg.thresholding = idx_thresholding_ratio
    self.idx_status_pub_callback()
    
  def update_sensor_thresholding(self, thresholding_ratio):
    # Sensor specific
    sensor_depth_confidence_val = int((float(self.SENSOR_MAX_THRESHOLD - self.SENSOR_MIN_THRESHOLD) * thresholding_ratio) + self.SENSOR_MIN_THRESHOLD)
    self.zed_dynamic_reconfig_client.update_configuration({"depth_confidence":sensor_depth_confidence_val})

  ### callback to get and apply range window controls
  def idx_set_range_window_callback(self, range_window_msg):
    rospy.loginfo(range_window_msg)
    
    self.idx_status_msg.range_window.start_range = range_window_msg.start_range
    self.idx_status_msg.range_window.stop_range = range_window_msg.stop_range  
    self.idx_status_pub_callback()


  #######################
  # Driver Data Publishers Functions

  ### callback to get and republish color 2d image
  def color_2d_image_callback(self, image_msg):
    # Publish to IDX namespace
    self.color_2d_image_msg = image_msg
    if not rospy.is_shutdown():
      self.idx_color_2d_image_pub.publish(image_msg)

  ### callback to get and republish bw 2d image
  def bw_2d_image_callback(self, image_msg):
    # Publish to IDX namespace
    if not rospy.is_shutdown():
      self.idx_bw_2d_image_pub.publish(image_msg)

  ### callback to get depthmap, republish it, convert it to global float array of meter depths corrisponding to image pixel location
  def depth_map_callback(self, depth_map_msg):
    # Zed depth data is floats in m, but passed as 4 bytes each that must be converted to floats
    # Use cv2_bridge() to convert the ROS image to OpenCV format
    #Convert the depth 4xbyte data to global float meter array
    cv2_bridge = CvBridge()
    cv2_depth_image = cv2_bridge.imgmsg_to_cv2(depth_map_msg, desired_encoding="passthrough")
    np_depth_array_m = (np.array(cv2_depth_image, dtype=np.float32)) # replace nan values
    np_depth_array_m[np.isnan(np_depth_array_m)] = 0
    ##################################################
    # Turn depth_array_m into colored image and publish
    # Create thresholded and 255 scaled version
    min_available_range = self.idx_status_msg.min_range_m 
    max_available_range = self.idx_status_msg.max_range_m
    range_span_m = max_available_range - min_available_range
    min_range_m=(self.idx_status_msg.range_window.start_range * (range_span_m)) + min_available_range
    max_range_m=(self.idx_status_msg.range_window.stop_range * (range_span_m)) + min_available_range
    np_depth_array_scaled = np_depth_array_m
    np_depth_array_scaled[np_depth_array_scaled < min_range_m] = 0
    np_depth_array_scaled[np_depth_array_scaled > max_range_m] = 0
    np_depth_array_scaled=np_depth_array_scaled-min_range_m
    max_value=np.max(np_depth_array_scaled)
    np_depth_array_scaled=np.array(np.abs(np_depth_array_scaled-float(max_value)),np.uint8) # Reverse for colormaping
    depth_scaler=max_range_m-min_range_m
    np_depth_array_scaled = np.array(255*np_depth_array_m/depth_scaler,np.uint8)
    ## Debug Code ###
    ##cv2.imwrite('/mnt/nepi_storage/data/image_bw.jpg', np_depth_array_scaled)
    ##print(np_depth_array_scaled.shape)
    ##time.sleep(1)
    #################
    # Apply colormap
    cv2_depth_image_color = cv2.applyColorMap(np_depth_array_scaled, cv2.COLORMAP_JET)
    ## Debug Code ###
    ##cv2.imwrite('/mnt/nepi_storage/data/image_color.jpg', im_color)
    ##print(im_color.shape)
    ##time.sleep(1)
    #################
    # Convert to cv2 image to Ros Image message
    ros_depth_image = cv2_bridge.cv2_to_imgmsg(cv2_depth_image_color,"bgr8")
    self.depth_image_msg = ros_depth_image
    # Publish new image to ros
    if not rospy.is_shutdown():
      self.idx_depth_map_pub.publish(depth_map_msg)
      self.idx_depth_image_pub.publish(ros_depth_image)

  ### callback to get and republish point_cloud and image
  def pointcloud_callback(self, pointcloud2_msg):
    # Publish to IDX namespace
    if not rospy.is_shutdown():
        self.idx_pointcloud_pub.publish(pointcloud2_msg)

  ### callback to provide capabilities report ###
  def idx_capabilities_query_callback(self, _):
    return self.idx_capabilities_report
  
  def __del__(self):
    rospy.loginfo("Shutting down: Executing script cleanup actions")
    # Unregister publishing topics
    self.idx_color_2d_image_pub.unregister()
    self.idx_bw_2d_image_pub.unregister()
    self.idx_depth_image_pub.unregister()
    self.idx_pointcloud_pub.unregister()
    self.idx_pointcloud_image_pub.unregister()

    self.zed_ros_wrapper_proc.terminate()


### Script Entrypoint
def startNode():
  rospy.loginfo("Starting ZED IDX node")
  rospy.init_node(name='zed')
    
  # Run initialization processes and start rospy.spin() via constructor
  node = ZedCameraNode()

#########################################
# Main
#########################################

if __name__ == '__main__':
  startNode()

