#!/usr/bin/env python
#
# Copyright (c) 2024 Numurus, LLC <https://www.numurus.com>.
#
# This file is part of nepi-engine
# (see https://github.com/nepi-engine).
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
#

import os
import sys
import time
import math
import rospy
import ros_numpy
import threading
import cv2
import copy


import subprocess
import dynamic_reconfigure.client
import numpy as np
import tf

from nepi_edge_sdk_base.idx_sensor_if import ROSIDXSensorIF

from nepi_edge_sdk_base import nepi_ros
from nepi_edge_sdk_base import nepi_nav
from nepi_edge_sdk_base import nepi_img
from nepi_edge_sdk_base import nepi_pc
from nepi_edge_sdk_base import nepi_nex

from datetime import datetime
from std_msgs.msg import UInt8, Empty, String, Bool, Float32
from sensor_msgs.msg import Image, PointCloud2
from nepi_ros_interfaces.msg import IDXStatus, RangeWindow, SaveDataStatus, SaveData, SaveDataRate
from nepi_ros_interfaces.srv import IDXCapabilitiesQuery, IDXCapabilitiesQueryResponse
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from dynamic_reconfigure.msg import Config
from rospy.numpy_msg import numpy_msg

class ZedCameraNode(object):
    DEFAULT_NODE_NAME = "zed_stereo_camera" # zed replaced with zed_type once discovered

    FACTORY_SETTINGS = nepi_nex.NONE_SETTINGS

    #Factory Control Values 
    FACTORY_CONTROLS = dict( controls_enable = True,
    auto_adjust = False,
    brightness_ratio = 0.5,
    contrast_ratio =  0.5,
    threshold_ratio =  0.0,
    resolution_mode = 3, # LOW, MED, HIGH, MAX
    framerate_mode = 3, # LOW, MED, HIGH, MAX
    start_range_ratio = 0.0, 
    stop_range_ratio = 1.0,
    min_range_m = 0.0,
    max_range_m = 20.0,
    zoom_ratio = 0.5, 
    rotate_ratio = 0.5,
    frame_id = 'nepi_center_frame' 
    )

    DEFAULT_CURRENT_FPS = 20 # Will be update later with actual

    ZED_MIN_RANGE_M_OVERRIDES = { 'zed': .2, 'zedm': .15, 'zed2': .2, 'zedx': .2} 
    ZED_MAX_RANGE_M_OVERRIDES = { 'zed':  15, 'zedm': 15, 'zed2': 20, 'zedx': 15} 

    # Render Image from pointcloud
    Render_Enable = True
    Render_Img_Width = 1280
    Render_Img_Height = 720
    Render_Background = [0, 0, 0, 0] # background color rgba
    Render_FOV = 60 # camera field of view in degrees
    Render_Center = [3, 0, 0]  # look_at target
    Render_Eye = [-5, -2.5, 0]  # camera position
    Render_Up = [0, 0, 1]  # camera orientation


    zed_type = 'zed'

    # Create shared class variables and thread locks 
    color_img_msg = None
    color_img_last_stamp = None
    color_img_lock = threading.Lock()
    bw_img_msg = None
    bw_img_last_stamp = None
    bw_img_lock = threading.Lock()
    depth_map_msg = None
    depth_map_last_stamp = None
    depth_map_lock = threading.Lock()   
    pc_msg = None
    pc_last_stamp = None
    pc_lock = threading.Lock()

    gps_msg = None
    odom_msg = None
    heading_msg = None

    def __init__(self):
  
        # This parameter should be automatically set by idx_sensor_mgr
        self.zed_type = rospy.get_param('~self.zed_type', 'zed2')

        # Initialize class variables
        self.factory_controls = self.FACTORY_CONTROLS

        # Apply OVERRIDES
        if self.zed_type in self.ZED_MIN_RANGE_M_OVERRIDES:
          self.factory_controls['min_range_m'] = self.ZED_MIN_RANGE_M_OVERRIDES[self.zed_type]
        if self.zed_type in self.ZED_MAX_RANGE_M_OVERRIDES:
          self.factory_controls['max_range_m'] = self.ZED_MAX_RANGE_M_OVERRIDES[self.zed_type]
        
        self.current_controls = self.factory_controls # Updateded during initialization
        self.current_fps = self.DEFAULT_CURRENT_FPS # Should be updateded when settings read

        self.caps_settings = nepi_nex.TEST_CAP_SETTINGS # None # Updateded during initialization
        self.factory_settings = nepi_nex.TEST_SETTINGS # None # Updateded during initialization
        self.current_settings = nepi_nex.TEST_SETTINGS # None # Updateded during initialization


        # Run the correct zed_ros_wrapper launch file
        zed_launchfile = self.zed_type + '.launch'
        zed_ros_wrapper_run_cmd = ['roslaunch', 'zed_wrapper', zed_launchfile]
        # TODO: Some process management for the Zed ROS wrapper
        self.zed_ros_wrapper_proc = subprocess.Popen(zed_ros_wrapper_run_cmd)

        # Connect to Zed node
        ZED_BASE_NAMESPACE = rospy.get_namespace() + self.zed_type + "/zed_node/"

        # Now that Zed SDK is started, we can set up the reconfig client
        self.zed_dynamic_reconfig_client = dynamic_reconfigure.client.Client(ZED_BASE_NAMESPACE, timeout=30)


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

        # Wait for zed camera topic to publish, then subscribe
        rospy.loginfo("Waiting for topic: " + ZED_COLOR_2D_IMAGE_TOPIC)
        nepi_ros.wait_for_topic(ZED_COLOR_2D_IMAGE_TOPIC)

        rospy.loginfo("Starting Zed IDX subscribers and publishers")
        rospy.Subscriber(ZED_COLOR_2D_IMAGE_TOPIC, Image, self.color_2d_image_callback, queue_size = 1)
        rospy.Subscriber(ZED_BW_2D_IMAGE_TOPIC, Image, self.bw_2d_image_callback, queue_size = 1)
        rospy.Subscriber(ZED_DEPTH_MAP_TOPIC, Image, self.depth_map_callback, queue_size = 1)
        rospy.Subscriber(ZED_POINTCLOUD_TOPIC, PointCloud2, self.pointcloud_callback, queue_size = 1)
        rospy.Subscriber(ZED_ODOM_TOPIC, Odometry, self.idx_odom_topic_callback)

        # Launch the ROS node
        rospy.loginfo("")
        rospy.loginfo("********************")
        rospy.loginfo("Starting " + self.DEFAULT_NODE_NAME.replace('zed',self.zed_type))
        rospy.loginfo("********************")
        rospy.loginfo("")
        rospy.init_node(self.DEFAULT_NODE_NAME) # Node name could be overridden via remapping
        self.node_name = rospy.get_name().split('/')[-1]
        rospy.loginfo(self.node_name + ": ... Connected!")


        idx_callback_names = {
            "Controls" : {
                # IDX Standard
                "Controls_Enable":  self.setControlsEnable,
                "Auto_Adjust":  self.setAutoAdjust,
                "Brightness": self.setBrightness,
                "Contrast":  self.setContrast,
                "Thresholding": self.setThresholding,
                "Resolution": self.setResolutionMode,
                "Framerate":  self.setFramerateMode,
                "Range": self.setRange,
                "Zoom": self.setZoom,
                "Rotate": None # self.setRotate
            },
            

            "Data" : {
                # Data callbacks
                "Color2DImg": self.getColorImg,
                "StopColor2DImg": self.stopColorImg,
                "BW2DImg": self.getBWImg,
                "StopBW2DImg": self.stopBWImg,
                "DepthMap": self.getDepthMap, 
                "StopDepthMap":  self.stopDepthMap,
                "DepthImg": self.getDepthImg, 
                "StopDepthImg":  self.stopDepthImg,
                "Pointcloud":  self.getPointcloud, 
                "StopPointcloud":  self.stopPointcloud,
                "PointcloudImg":  self.getPointcloudImg, 
                "StopPointcloudImg":  self.stopPointcloudImg,
                "GPS": None,
                "Odom": self.getOdom,
                "Heading": None,
            }
        }

        # IDX Remappings
        # Now that we've initialized the callbacks table, can apply the remappings
        idx_remappings = rospy.get_param('~idx_remappings', {})
        rospy.loginfo(self.node_name + ': Establishing IDX remappings')
        for from_name in idx_remappings:
            to_name = idx_remappings[from_name]
            if (from_name not in idx_callback_names["Controls"]) and (from_name not in idx_callback_names["Data"]):
                rospy.logwarn('\tInvalid IDX remapping target: ' + from_name)
            elif from_name in idx_callback_names["Controls"]:
                if self.driver.hasAdjustableCameraControl(idx_remappings[to_name]) is False:
                    rospy.logwarn('\tRemapping ' + from_name + ' to an unavailable control (' + to_name + ')')
                else:
                    rospy.loginfo('\t' + from_name + '-->' + to_name)
                    idx_callback_names["Controls"][from_name] = lambda x: self.setDriverCameraControl(to_name, x)
            elif (from_name in idx_callback_names["Controls"]):
                # if (TODO: check data availability from driver):
                #    rospy.logwarn('\tRemapping ' + from_name + ' to an unavailable data source (' + to_name + ')')
                
                # For now, this is unsupported
                rospy.logwarn('\tRemapping IDX data for V4L2 devices not yet supported')
            else:
                idx_callback_names[from_name] = idx_callback_names[to_name]
                rospy.loginfo('\t' + from_name + '-->' + to_name)


        # Initialize controls and settings variables
        self.factory_controls = self.FACTORY_CONTROLS
        self.current_controls = self.factory_controls # Updateded during initialization
        self.current_fps = self.DEFAULT_CURRENT_FPS # Should be updateded when settings read
        self.caps_settings = self.getCapSettings() 
        self.factory_settings = self.FACTORY_SETTINGS

        # Launch the IDX interface --  this takes care of initializing all the camera settings from config. file
        rospy.loginfo(self.node_name + ": Launching NEPI IDX (ROS) interface...")
        self.idx_if = ROSIDXSensorIF(sensor_name=self.node_name,
                                     factorySettings = self.factory_settings,
                                     settingsUpdateFunction=self.settingsUpdateFunction,
                                     getSettings=self.getSettings,
                                     factoryControls = self.FACTORY_CONTROLS,
                                     setControlsEnable = idx_callback_names["Controls"]["Controls_Enable"],
                                     setAutoAdjust= idx_callback_names["Controls"]["Auto_Adjust"],
                                     setResolutionMode=idx_callback_names["Controls"]["Resolution"], 
                                     setFramerateMode=idx_callback_names["Controls"]["Framerate"], 
                                     setContrast=idx_callback_names["Controls"]["Contrast"], 
                                     setBrightness=idx_callback_names["Controls"]["Brightness"], 
                                     setThresholding=idx_callback_names["Controls"]["Thresholding"], 
                                     setRange=idx_callback_names["Controls"]["Range"], 
                                     setZoom=idx_callback_names["Controls"]["Zoom"], 
                                     setRotate=idx_callback_names["Controls"]["Rotate"], 
                                     getColor2DImg=idx_callback_names["Data"]["Color2DImg"], 
                                     stopColor2DImgAcquisition=idx_callback_names["Data"]["StopColor2DImg"],
                                     getBW2DImg=idx_callback_names["Data"]["BW2DImg"], 
                                     stopBW2DImgAcquisition=idx_callback_names["Data"]["StopBW2DImg"],
                                     getDepthMap=idx_callback_names["Data"]["DepthMap"], 
                                     stopDepthMapAcquisition=idx_callback_names["Data"]["StopDepthMap"],
                                     getDepthImg=idx_callback_names["Data"]["DepthImg"], 
                                     stopDepthImgAcquisition=idx_callback_names["Data"]["StopDepthImg"],
                                     getPointcloud=idx_callback_names["Data"]["Pointcloud"], 
                                     stopPointcloudAcquisition=idx_callback_names["Data"]["StopPointcloud"],
                                     getPointcloudImg=idx_callback_names["Data"]["PointcloudImg"], 
                                     stopPointcloudImgAcquisition=idx_callback_names["Data"]["StopPointcloudImg"],
                                     getGPSMsg=idx_callback_names["Data"]["GPS"],
                                     getOdomMsg=idx_callback_names["Data"]["Odom"],
                                     getHeadingMsg=idx_callback_names["Data"]["Heading"])
        rospy.loginfo(self.node_name + ": ... IDX interface running")

        # Update available IDX callbacks based on capabilities that the driver reports
        self.logDeviceInfo()

        # Now that all camera start-up stuff is processed, we can update the camera from the parameters that have been established
        self.idx_if.updateFromParamServer()

        # Now start the node
        rospy.spin()

    #**********************
    # Zed camera data callbacks

    # callback to get color 2d image data
    def color_2d_image_callback(self, image_msg):
      if self.color_img_lock.locked() is False:
        self.color_img_lock.acquire()
        self.color_img_msg = image_msg
        self.color_img_lock.release()
      else:
        pass # skip this msg to ensure latest is cached when ready

    # callback to get 2d image data
    def bw_2d_image_callback(self, image_msg):
      if self.bw_img_lock.locked() is False:
        self.bw_img_lock.acquire()
        self.bw_img_msg = image_msg
        self.bw_img_lock.release()
      else:
        pass # skip this msg to ensure latest is cached when ready

    # callback to get depthmap
    def depth_map_callback(self, image_msg):
      if self.depth_map_lock.locked() is False:
        self.depth_map_lock.acquire()
        self.depth_map_msg = image_msg
        self.depth_map_lock.release()
      else:
        pass # skip this msg to ensure latest is cached when ready
      self.depth_map_msg = image_msg


    # callback to get and republish point_cloud and image
    def pointcloud_callback(self, pointcloud_msg):
      if self.pc_lock.locked() is False:
        self.pc_lock.acquire()
        self.pc_msg = pointcloud_msg
        self.pc_lock.release()
      else:
        pass # skip this msg to ensure latest is cached when ready

    # Callback to get odom data
    def idx_odom_topic_callback(self, odom_msg):
      self.odom_msg = odom_msg


    #**********************
    # IDX driver functions

    def logDeviceInfo(self):
        device_info_str = self.node_name + " info:\n"
        rospy.loginfo(device_info_str)

    def getCapSettings(self):
        cap_settings = nepi_nex.NONE_SETTINGS
        # Replace with get cap settings process
        return cap_settings

    def settingsUpdateFunction(self,setting):
        success = False
        # Add update setting process
        success = True
        return success
    
    def getSettings(self):
        settings = nepi_nex.NONE_SETTINGS
        # Replace with get settings process
        return settings


    def setControlsEnable(self, enable):
        self.current_controls["controls_enable"] = enable
        status = True
        err_str = ""
        return status, err_str
        
    def setAutoAdjust(self, enable):
        ret = self.current_controls["auto_adjust"] = enable
        status = True
        err_str = ""
        return status, err_str

    def setBrightness(self, ratio):
        if ratio > 1:
            ratio = 1
        elif ratio < 0:
            ratio = 0
        self.current_controls["brightness_ratio"] = ratio
        status = True
        err_str = ""
        return status, err_str

    def setContrast(self, ratio):
        if ratio > 1:
            ratio = 1
        elif ratio < 0:
            ratio = 0
        self.current_controls["contrast_ratio"] = ratio
        status = True
        err_str = ""
        return status, err_str

    def setThresholding(self, ratio):
        if ratio > 1:
            ratio = 1
        elif ratio < 0:
            ratio = 0
        self.current_controls["threshold_ratio"] = ratio
        status = True
        err_str = ""
        return status, err_str

    def setResolutionMode(self, mode):
        if (mode > self.idx_if.RESOLUTION_MODE_MAX):
            return False, "Invalid mode value"
        self.current_controls["resolution_mode"] = mode
        status = True
        err_str = ""
        return status, err_str
    
    def setFramerateMode(self, mode):
        if (mode > self.idx_if.FRAMERATE_MODE_MAX):
            return False, "Invalid mode value"
        self.current_controls["framerate_mode"] = mode
        status = True
        err_str = ""
        return status, err_str

    def setRange(self, min_ratio, max_ratio):
        if min_ratio > 1:
            min_ratio = 1
        elif min_ratio < 0:
            min_ratio = 0
        self.current_controls["start_range_ratio"] = min_ratio
        if max_ratio > 1:
            max_ratio = 1
        elif max_ratio < 0:
            max_ratio = 0
        if min_ratio < max_ratio:
          self.current_controls["stop_range_ratio"] = max_ratio
          status = True
          err_str = ""
        else:
          status = False
          err_str = "Invalid Range Window"
        return status, err_str

    def setZoom(self, ratio):
        if ratio > 1:
            ratio = 1
        elif ratio < 0:
            ratio = 0
        self.current_controls["zoom_ratio"] = ratio
        status = True
        err_str = ""
        return status, err_str

    def setRotate(self, ratio):
        if ratio > 1:
            ratio = 1
        elif ratio < 0:
            ratio = 0
        self.current_controls["rotate_ratio"] = ratio
        status = True
        err_str = ""
        return status, err_str

    def setDriverCameraControl(self, control_name, value):
        pass # Need to implement
        #return self.driver.setScaledCameraControl(control_name, value)
    

    # Good base class candidate - Shared with ONVIF
    def getColorImg(self):
        # Set process input variables
        data_product = "color_2d_image"
        img_msg = self.color_img_msg
        img_last_stamp = self.color_img_last_stamp
        lock = self.color_img_lock
        encoding = 'rgb8'

        # Run get process
        # Initialize some process return variables
        status = False
        msg = ""
        ros_img = None
        cv2_img = None
        ros_timestamp = None
        if img_msg is not None:
          if img_msg.header.stamp != img_last_stamp:
            lock.acquire()
            ros_img = img_msg
            ros_timestamp = img_msg.header.stamp
            status = True
            msg = ""
            ros_timestamp = ros_img.header.stamp
            if self.current_controls.get("controls_enable"):
              cv2_img =  nepi_img.rosimg_to_cv2img(ros_img, encoding = encoding)
              cv2_img = nepi_nex.applyIDXControls2Image(cv2_img,self.current_controls,self.current_fps)
              #ros_img = nepi_img.cv2img_to_rosimg(cv2_img, encoding = encoding)
            img_last_stamp = ros_timestamp
            lock.release()
          else:
            msg = "No new data for " + data_product + " available"
        else:
          msg = "Received None type data for " + data_product + " process"
        if cv2_img is not None:
          return status, msg, cv2_img, ros_timestamp, encoding
        else: 
          return status, msg, ros_img, ros_timestamp, encoding
    
    # Good base class candidate - Shared with ONVIF
    def stopColorImg(self):
        ret = True
        msg = "Success"
        return ret,msg
    
    # Good base class candidate - Shared with ONVIF
    def getBWImg(self):
        # Set process input variables
        data_product = "bw_2d_image"
        img_msg = self.bw_img_msg
        img_last_stamp = self.bw_img_last_stamp
        lock = self.bw_img_lock
        encoding = "mono8"

        # Run get process
        # Initialize some process return variables
        status = False
        msg = ""
        ros_img = None
        cv2_img = None
        ros_timestamp = None
        if img_msg is not None:
          if img_msg.header.stamp != img_last_stamp:
            lock.acquire()
            ros_img = img_msg
            ros_timestamp = img_msg.header.stamp
            status = True
            msg = ""
            ros_timestamp = ros_img.header.stamp
            if self.current_controls.get("controls_enable"):
              cv2_img =  nepi_img.rosimg_to_cv2img(ros_img, encoding = encoding)
              cv2_img = nepi_nex.applyIDXControls2Image(cv2_img,self.current_controls,self.current_fps)
              #ros_img = nepi_img.cv2img_to_rosimg(cv2_img, encoding = encoding)
            img_last_stamp = ros_timestamp
            lock.release()
          else:
            msg = "No new data for " + data_product + " available"
        else:
          msg = "Received None type data for " + data_product + " process"
        if cv2_img is not None:
          return status, msg, cv2_img, ros_timestamp, encoding
        else: 
          return status, msg, ros_img, ros_timestamp, encoding
    
    # Good base class candidate - Shared with ONVIF
    def stopBWImg(self):
        ret = True
        msg = "Success"
        return ret,msg

    def getDepthMap(self):
        # Set process input variables
        data_product = "depth_map"
        img_msg = self.depth_map_msg
        img_last_stamp = self.depth_map_last_stamp
        lock = self.depth_map_lock
        encoding = '32FC1'
        # Run get process
        # Initialize some process return variables
        status = False
        msg = ""
        cv2_img = None
        ros_img = None
        ros_timestamp = None
        if img_msg is not None:
          if img_msg.header.stamp != img_last_stamp:
            lock.acquire()
            status = True
            msg = ""
            ros_img = copy.deepcopy(img_msg)
            ros_timestamp = ros_img.header.stamp
            img_last_stamp = ros_timestamp
            lock.release()

            # Convert ros depth_map to cv2_img and numpy depth data
            cv2_depth_map = nepi_img.rosimg_to_cv2img(ros_img, encoding="passthrough")
            depth_data = (np.array(cv2_depth_map, dtype=np.float32)) # replace nan values
            # Get range data
            start_range_ratio = self.current_controls.get("start_range_ratio")
            stop_range_ratio = self.current_controls.get("stop_range_ratio")
            min_range_m = self.current_controls.get("min_range_m")
            max_range_m = self.current_controls.get("max_range_m")
            delta_range_m = max_range_m - min_range_m
            # Adjust range Limits if IDX Controls enabled and range ratios are not min/max
            if self.current_controls.get("controls_enable") and (start_range_ratio > 0 or stop_range_ratio < 1):
              max_range_m = min_range_m + stop_range_ratio * delta_range_m
              min_range_m = min_range_m + start_range_ratio * delta_range_m
              delta_range_m = max_range_m - min_range_m
              rospy.loginfo("updated range values")
            # Filter depth_data in range
            depth_data[np.isnan(depth_data)] = float('nan')  # set to NaN
            depth_data[depth_data <= min_range_m] = float('nan')  # set to NaN
            depth_data[depth_data >= max_range_m] = float('nan')  # set to NaN
            cv2_img = depth_data
            #ros_img = nepi_img.cv2img_to_rosimg(cv2_depth_image,encoding)
           
          else:
            msg = "No new data for " + data_product + " available"
        else:
          msg = "Received None type data for " + data_product + " process"
        if cv2_img is not None:
          return status, msg, cv2_img, ros_timestamp, encoding
        else: 
          return status, msg, ros_img, ros_timestamp, encoding
    
    def stopDepthMap(self):
        ret = True
        msg = "Success"
        return ret,msg


    def getDepthImg(self):
        # Set process input variables
        data_product = "depth_image"
        img_msg = self.depth_map_msg
        img_last_stamp = self.depth_map_last_stamp
        lock = self.depth_map_lock
        encoding = 'bgr8'
         # Run get process
        # Initialize some process return variables
        status = False
        msg = ""
        cv2_img = None
        ros_img = None
        ros_timestamp = None
        if img_msg is not None:
          if img_msg.header.stamp != img_last_stamp:
            lock.acquire()
            status = True
            msg = ""
            ros_img = copy.deepcopy(img_msg)
            ros_timestamp = ros_img.header.stamp
            img_last_stamp = ros_timestamp
            lock.release()

            # Convert ros depth_map to cv2_img and numpy depth data
            cv2_depth_map = nepi_img.rosimg_to_cv2img(ros_img, encoding="passthrough")
            depth_data = (np.array(cv2_depth_map, dtype=np.float32)) # replace nan values
            # Get range data
            start_range_ratio = self.current_controls.get("start_range_ratio")
            stop_range_ratio = self.current_controls.get("stop_range_ratio")
            min_range_m = self.current_controls.get("min_range_m")
            max_range_m = self.current_controls.get("max_range_m")
            delta_range_m = max_range_m - min_range_m
            # Adjust range Limits if IDX Controls enabled and range ratios are not min/max
            if self.current_controls.get("controls_enable") and (start_range_ratio > 0 or stop_range_ratio < 1):
              max_range_m = min_range_m + stop_range_ratio * delta_range_m
              min_range_m = min_range_m + start_range_ratio * delta_range_m
              delta_range_m = max_range_m - min_range_m
            # Filter depth_data in range
            depth_data[np.isnan(depth_data)] = max_range_m 
            depth_data[depth_data <= min_range_m] = max_range_m # set to max
            depth_data[depth_data >= max_range_m] = max_range_m # set to max
            # Create colored cv2 depth image
            depth_data = depth_data - min_range_m # Shift down 
            depth_data = np.abs(depth_data - max_range_m) # Reverse for colormap
            depth_data = np.array(255*depth_data/delta_range_m,np.uint8) # Scale for bgr colormap
            cv2_img = cv2.applyColorMap(depth_data, cv2.COLORMAP_JET)
            #ros_img = nepi_img.cv2img_to_rosimg(cv2_depth_image,encoding)
          else:
            msg = "No new data for " + data_product + " available"
        else:
          msg = "Received None type data for " + data_product + " process"
        if cv2_img is not None:
          return status, msg, cv2_img, ros_timestamp, encoding
        else: 
          return status, msg, ros_img, ros_timestamp, encoding

    
    def stopDepthImg(self):
        ret = True
        msg = "Success"
        return ret,msg

    def getPointcloud(self):     
        # Set process input variables
        data_product = "pointcloud"
        pc_msg = self.pc_msg
        pc_last_stamp = self.pc_last_stamp
        lock = self.pc_lock
        # Run get process
        # Initialize some process return variables
        status = False
        msg = ""
        ros_pc = None
        o3d_pc = None
        ros_timestamp = None
        ros_frame = None
        if pc_msg is not None:
          if pc_msg.header.stamp != pc_last_stamp:
            lock.acquire()
            ros_pc = copy.deepcopy(pc_msg)
            ros_timestamp = pc_msg.header.stamp
            ros_frame = pc_msg.header.frame_id
            status = True
            msg = ""
            pc_last_stamp = ros_timestamp
            if self.current_controls.get("controls_enable"):
              start_range_ratio = self.current_controls.get("start_range_ratio")
              stop_range_ratio = self.current_controls.get("stop_range_ratio")
              min_range_m = self.current_controls.get("min_range_m")
              max_range_m = self.current_controls.get("max_range_m")
              delta_range_m = max_range_m - min_range_m
              range_clip_min_range_m = min_range_m + start_range_ratio  * delta_range_m
              range_clip_max_range_m = min_range_m + stop_range_ratio  * delta_range_m
              if start_range_ratio > 0 or stop_range_ratio < 1:
                o3d_pc = nepi_pc.rospc_to_o3dpc(ros_pc, remove_nans=False)
                o3d_pc = nepi_pc.range_clip( o3d_pc, range_clip_min_range_m, range_clip_max_range_m)
          else:
            msg = "No new data for " + data_product + " available"
        else:
          msg = "Received None type data for " + data_product + " process"
        lock.release()
        if o3d_pc is not None:
          return status, msg, o3d_pc, ros_timestamp, ros_frame
        else: 
          return status, msg, ros_pc, ros_timestamp, ros_frame

    
    def stopPointcloud(self):
        ret = True
        msg = "Success"
        return ret,msg

    def getPointcloudImg(self):
        data_product = "pointcloud_image"
        pc_msg = self.pc_msg
        pc_last_stamp = self.pc_last_stamp
        lock = self.pc_lock
        encoding = 'bgr8'
         # Run get process
        # Initialize some process return variables
        status = False
        msg = ""
        ros_img = None
        ros_timestamp = None
        ros_frame = None
        if pc_msg is not None:
          if pc_msg.header.stamp != pc_last_stamp:
            lock.acquire()
            status = True
            msg = ""
            o3d_pc = nepi_pc.rospc_to_o3dpc(pc_msg, remove_nans=True)
            ros_timestamp = pc_msg.header.stamp
            ros_frame = pc_msg.header.frame_id
            pc_last_stamp = ros_timestamp
            lock.release()
            
            if self.current_controls.get("controls_enable"):
              # Range Clip Pointcloud
              start_range_ratio = self.current_controls.get("start_range_ratio")
              stop_range_ratio = self.current_controls.get("stop_range_ratio")
              min_range_m = self.current_controls.get("min_range_m")
              max_range_m = self.current_controls.get("max_range_m")
              delta_range_m = max_range_m - min_range_m
              range_clip_min_range_m = min_range_m + start_range_ratio  * delta_range_m
              range_clip_max_range_m = min_range_m + stop_range_ratio  * delta_range_m
              if start_range_ratio > 0 or stop_range_ratio < 1:
                o3d_pc = nepi_pc.range_clip( o3d_pc, range_clip_min_range_m, range_clip_max_range_m)
              # Adjust resolution  and downsample for reduced rendering time
              img_width = self.Render_Img_Width
              img_height = self.Render_Img_Height
              
              #res_scaler = float((self.current_controls.get("resolution_mode")) + 1) / float(4)
              #img_width = self.Render_Img_Width # int(self.Render_Img_Width * res_scaler)
              #img_height = self.Render_Img_Height # int(self.Render_Img_Height * res_scaler)
              #k_points = int(100 * float(self.current_controls.get("resolution_mode"))/float(4)**2)
              #o3d_pc = nepi_pc.uniform_down_sampling(o3d_pc, every_k_points = k_points)
              
              # Adjust Zoom for rendering
              zoom_scaler = 1 - self.current_controls.get("zoom_ratio")
              render_eye = [number*zoom_scaler for number in self.Render_Eye] # Apply IDX zoom control

              # Adjust Rotate for rendering
              #hor_vector = [render_eye[0],render_eye[1]]
              #hor_rot_ratio = self.current_controls.get("rotate_ratio")
              #hor_vector_rot = nepi_pc.vector_rotate(hor_vector,hor_rot_ratio) 
              #render_eye[0] = hor_vector_rot[0]
              #render_eye[1] = hor_vector_rot[1]
              #rospy.loginfo(render_eye)

            else: 
              img_width = self.Render_Img_Width
              img_height = self.Render_Img_Height
              render_eye = self.Render_Eye
            if not rospy.is_shutdown():
              o3d_img = nepi_pc.render_image(o3d_pc,img_width,img_height,
                          self.Render_Background,self.Render_FOV,self.Render_Center,render_eye,self.Render_Up)
              ros_img = nepi_pc.o3dimg_to_rosimg(o3d_img, stamp=ros_timestamp, frame_id=ros_frame)
            
          else:
            msg = "No new data for " + data_product + " available"
        else:
          msg = "Received None type data for " + data_product + " process"
        return status, msg, ros_img, ros_timestamp, encoding
    
    def stopPointcloudImg(self):
        ret = True
        msg = "Success"
        return ret,msg

    def getOdom(self):
        return self.odom_msg
        
if __name__ == '__main__':
    node = ZedCameraNode()
