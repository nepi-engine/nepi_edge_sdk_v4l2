#!/usr/bin/env python
#
# Copyright (c) 2024 Numurus, LLC <https://www.numurus.com>.
#
# This file is part of nepi-engine
# (see https://github.com/nepi-engine).
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
#
import sys
import time
import math
import rospy
import threading
import cv2

from nepi_edge_sdk_base.idx_sensor_if import ROSIDXSensorIF
from nepi_edge_sdk_v4l2.v4l2_cam_driver import V4L2_GENERIC_DRIVER_ID, V4l2CamDriver

from nepi_edge_sdk_base import nepi_img
from nepi_edge_sdk_base import nepi_nex

class V4l2CameraNode:
    DEFAULT_NODE_NAME = "v4l2_camera_node"

    #Factory Control Values 
    FACTORY_CONTROLS = dict( controls_enable = True,
    auto_adjust = False,
    brightness_ratio = 0.5,
    contrast_ratio =  0.5,
    threshold_ratio =  0.0,
    resolution_mode = 3, # LOW, MED, HIGH, MAX
    framerate_mode = 3, # LOW, MED, HIGH, MAX
    start_range_ratio = 0.0, 
    stop_ranage_ratio = 1.0,
    min_range_m = 0.0,
    max_range_m = 1.0 
    )
    # Define class variables
    factory_controls = FACTORY_CONTROLS
    current_controls = FACTORY_CONTROLS # Updateded during initialization
    current_fps = 20 # Updateded during initialization

    caps_settings = nepi_nex.TEST_CAP_SETTINGS # None # Updateded during initialization
    factory_settings = nepi_nex.TEST_SETTINGS # None # Updateded during initialization
    current_settings = nepi_nex.TEST_SETTINGS # None # Updateded during initialization
 
    DRIVER_SPECIALIZATION_CONSTRUCTORS = {V4L2_GENERIC_DRIVER_ID: V4l2CamDriver} # Only the generic on at the moment

    def __init__(self):
        # Launch the ROS node
        rospy.loginfo("Starting " + self.DEFAULT_NODE_NAME)
        rospy.init_node(self.DEFAULT_NODE_NAME) # Node name could be overridden via remapping
        self.node_name = rospy.get_name().split('/')[-1]

        device_path = rospy.get_param('~device_path', default='/dev/video0')

        # Set up for specialized drivers here
        self.driver_id = rospy.get_param('~driver_id', V4L2_GENERIC_DRIVER_ID)
        rospy.set_param('~driver_id', self.driver_id)
        if self.driver_id not in self.DRIVER_SPECIALIZATION_CONSTRUCTORS:
            rospy.logerr(self.node_name + ": unknown driver_id " + self.driver_id)
            return
        DriverConstructor = self.DRIVER_SPECIALIZATION_CONSTRUCTORS[self.driver_id]

        # Start the driver to connect to the camera
        rospy.loginfo(self.node_name + ": Launching " + self.driver_id + " driver")
        try:
            self.driver = DriverConstructor(device_path)
        except Exception as e:
            # Only log the error every 30 seconds -- don't want to fill up log in the case that the camera simply isn't attached.
            rospy.logerr(self.node_name + ": Failed to instantiate driver - " + str(e) + ")")
            sys.exit(-1)

        if not self.driver.isConnected():
            rospy.logerr(self.node_name + ": Failed to connect to camera device")
            
        rospy.loginfo(self.node_name + ": ... Connected!")

        self.createResolutionModeMapping()
        self.createFramerateModeMapping()

        idx_callback_names = {
            "Controls" : {
                # IDX Standard
                # Mode controls may need some intervention
                "Controls_Enable":  self.setControlsEnable,
                "Auto_Adjust":  self.setAutoAdjust,
                "Brightness": self.setBrightness,
                "Contrast":  self.setContrast,
                "Thresholding": self.setThresholding,
                "Resolution": self.setResolutionMode,
                "Framerate":  self.setFramerateMode,
                "Range":  None
            },
            

            "Data" : {
                # Data callbacks
                "Color2DImg": self.getColorImg,
                "StopColor2DImg": self.stopColorImg,
                "BW2DImg": self.getBWImg,
                "StopBW2DImg": self.stopBWImg,
                # Following have no driver support, can be remapped though
                "DepthMap": None, 
                "StopDepthMap": None,
                "DepthImg": None, 
                "StopDepthImg": None,
                "Pointcloud": None, 
                "StopPointcloud": None,
                "PointcloudImg": None, 
                "StopPointcloudImg": None
            }
        }

        # IDX Remappings
        # Now that we've initialized the callbacks table, can apply the remappings
##        idx_remappings = rospy.get_param('~idx_remappings', {})
##        rospy.loginfo(self.node_name + ': Establishing IDX remappings')
##        for from_name in idx_remappings:
##            to_name = idx_remappings[from_name]
##            if (from_name not in idx_callback_names["Controls"]) and (from_name not in idx_callback_names["Data"]):
##                rospy.logwarn('\tInvalid IDX remapping target: ' + from_name)
##            elif from_name in idx_callback_names["Controls"]:
##                if self.driver.hasAdjustableCameraControl(idx_remappings[to_name]) is False:
##                    rospy.logwarn('\tRemapping ' + from_name + ' to an unavailable control (' + to_name + ')')
##                else:
##                    rospy.loginfo('\t' + from_name + '-->' + to_name)
##                    idx_callback_names["Controls"][from_name] = lambda x: self.setDriverCameraControl(to_name, x)
##            elif (from_name in idx_callback_names["Controls"]):
##                # if (TODO: check data availability from driver):
##                #    rospy.logwarn('\tRemapping ' + from_name + ' to an unavailable data source (' + to_name + ')')
##                
##                # For now, this is unsupported
##                rospy.logwarn('\tRemapping IDX data for V4L2 devices not yet supported')
##            else:
##                idx_callback_names[from_name] = idx_callback_names[to_name]
##                rospy.loginfo('\t' + from_name + '-->' + to_name)

        # Create threading locks, controls, and status
        self.img_lock = threading.Lock()
        self.color_image_acquisition_running = False
        self.bw_image_acquisition_running = False
        self.cached_2d_color_frame = None
        self.cached_2d_color_frame_timestamp = None

        # Launch the IDX interface --  this takes care of initializing all the camera settings from config. file
        rospy.loginfo(self.node_name + ": Launching NEPI IDX (ROS) interface...")
        self.idx_if = ROSIDXSensorIF(sensor_name=self.node_name,
                                     factoryControls = self.FACTORY_CONTROLS,
                                     setControlsEnable = idx_callback_names["Controls"]["Controls_Enable"],
                                     setAutoAdjust= idx_callback_names["Controls"]["Auto_Adjust"],
                                     setResolutionMode=idx_callback_names["Controls"]["Resolution"], 
                                     setFramerateMode=idx_callback_names["Controls"]["Framerate"], 
                                     setContrast=idx_callback_names["Controls"]["Contrast"], 
                                     setBrightness=idx_callback_names["Controls"]["Brightness"], 
                                     setThresholding=idx_callback_names["Controls"]["Thresholding"], 
                                     setRange=idx_callback_names["Controls"]["Range"], 
                                     getColor2DImg=idx_callback_names["Data"]["Color2DImg"], 
                                     stopColor2DImgAcquisition=idx_callback_names["Data"]["StopColor2DImg"],
                                     getGrayscale2DImg=idx_callback_names["Data"]["BW2DImg"], 
                                     stopGrayscale2DImgAcquisition=idx_callback_names["Data"]["StopBW2DImg"],
                                     getDepthMap=idx_callback_names["Data"]["DepthMap"], 
                                     stopDepthMapAcquisition=idx_callback_names["Data"]["StopDepthMap"],
                                     getDepthImg=idx_callback_names["Data"]["DepthImg"], 
                                     stopDepthImgAcquisition=idx_callback_names["Data"]["StopDepthImg"],
                                     getPointcloud=idx_callback_names["Data"]["Pointcloud"], 
                                     stopPointcloudAcquisition=idx_callback_names["Data"]["StopPointcloud"],
                                     getPointcloudImg=idx_callback_names["Data"]["PointcloudImg"], 
                                     stopPointcloudImgAcquisition=idx_callback_names["Data"]["StopPointcloudImg"])
        rospy.loginfo(self.node_name + ": ... IDX interface running")

        # Update available IDX callbacks based on capabilities that the driver reports
        self.logDeviceInfo()

        # Now that all camera start-up stuff is processed, we can update the camera from the parameters that have been established
        self.idx_if.updateFromParamServer()

        # Now start the node
        rospy.spin()

    def logDeviceInfo(self):
        device_info_str = self.node_name + " info:\n"
        device_info_str += "\tDevice Path: " + self.driver.device_path + "\n"

        scalable_cam_controls = self.driver.getAvailableScaledCameraControls()
        discrete_cam_controls = self.driver.getAvailableDiscreteCameraControls()
        device_info_str += "\tCamera Controls:\n"
        for ctl in scalable_cam_controls:
            device_info_str += ("\t\t" + ctl + "\n")
        for ctl in discrete_cam_controls:
            device_info_str += ("\t\t" + ctl + ': ' + str(discrete_cam_controls[ctl]) + "\n")
        
        _, format = self.driver.getCurrentFormat()
        device_info_str += "\tCamera Output Format: " + format + "\n"

        _, resolution_dict = self.driver.getCurrentResolution()
        device_info_str += "\tCurrent Resolution: " + str(resolution_dict['width']) + 'x' + str(resolution_dict['height']) + "\n"

        if (self.driver.hasAdjustableResolution()):
            _, available_resolutions = self.driver.getCurrentFormatAvailableResolutions()
            device_info_str += "\tAvailable Resolutions:\n"
            for res in available_resolutions:
                device_info_str += "\t\t" + str(res["width"]) + 'x' + str(res["height"]) + "\n"

        if (self.driver.hasAdjustableFramerate()):
            _, available_framerates = self.driver.getCurrentResolutionAvailableFramerates()
            device_info_str += "\tAvailable Framerates (current resolution): " + str(available_framerates) + "\n"

        device_info_str += "\tResolution Modes:\n"
        for mode in self.resolution_mode_map:
            device_info_str += "\t\t" + str(mode) + ': ' + str(self.resolution_mode_map[mode]['width']) + 'x' + str(self.resolution_mode_map[mode]['height']) + "\n"
        
        device_info_str += "\tFramerate Modes (current resolution):\n"
        for mode in self.framerate_mode_map:
            device_info_str += "\t\t" + str(mode) + ': ' + str(self.framerate_mode_map[mode]) + "\n"
        
        rospy.loginfo(device_info_str)

    def createResolutionModeMapping(self):
        _, available_resolutions = self.driver.getCurrentFormatAvailableResolutions()
        available_resolution_count = len(available_resolutions) 
        # Check if this camera supports resolution adjustment
        if (available_resolution_count == 0):
            self.resolution_mode_map = {}
            return
        
        #available_resolutions is a list of dicts, sorted by "width" from smallest to largest
        # Distribute the modes evenly
        resolution_mode_count = ROSIDXSensorIF.RESOLUTION_MODE_MAX + 1
        # Ensure the highest resolution is available as "Ultra", others are spread evenly amongst remaining options
        self.resolution_mode_map = {resolution_mode_count - 1:available_resolutions[available_resolution_count - 1]}

        resolution_step = int(math.floor(available_resolution_count / resolution_mode_count))
        if resolution_step == 0:
            resolution_step = 1
        
        for i in range(1,resolution_mode_count):
            res_index = (available_resolution_count - 1) - (i*resolution_step)
            if res_index < 0:
                res_index = 0
            self.resolution_mode_map[resolution_mode_count - i - 1] = available_resolutions[res_index]

    def createFramerateModeMapping(self):
        _, available_framerates = self.driver.getCurrentResolutionAvailableFramerates()
        
        available_framerate_count = len(available_framerates)
        if (available_framerate_count == 0):
            self.framerate_mode_map = {}
            return
        
        #rospy.loginfo("Debug: Creating Framerate Mode Mapping")

        framerate_mode_count = ROSIDXSensorIF.FRAMERATE_MODE_MAX + 1
        # Ensure the highest framerate is available as "Ultra", others are spread evenly amongst remaining options
        self.framerate_mode_map = {framerate_mode_count - 1: available_framerates[available_framerate_count - 1]}
        
        framerate_step = int(math.floor(available_framerate_count / framerate_mode_count))
        if framerate_step == 0:
            framerate_step = 1

        for i in range(1, framerate_mode_count):
            framerate_index = (available_framerate_count - 1) - (i*framerate_step)
            if framerate_index < 0:
                framerate_index = 0
            self.framerate_mode_map[framerate_mode_count - i - 1] = available_framerates[framerate_index]


    
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
        if (mode >= len(self.resolution_mode_map)):
            return False, "Invalid mode value"
        self.current_controls["resolution_mode"] = mode
        status = True
        err_str = ""
        return status, err_str
    
    def setFramerateMode(self, mode):
        if (mode >= len(self.framerate_mode_map)):
            return False, "Invalid mode value"
        self.current_controls["framerate_mode"] = mode
        status = True
        err_str = ""
        return status, err_str

    def applyControls2Frame(self,frame):
        if self.current_controls.get("controls_enable"): 
            resolution_ratio = self.current_controls.get("resolution_mode")/3
            [frame,new_res] = nepi_img.adjust_resolution(frame, resolution_ratio)
            if self.current_controls.get("auto_adjust") is False:
                frame = nepi_img.adjust_brightness(frame,self.current_controls.get("brightness_ratio"))
                
                frame = nepi_img.adjust_contrast(frame,self.current_controls.get("contrast_ratio"))
                frame = nepi_img.adjust_sharpness(frame,self.current_controls.get("threshold_ratio"))
            else:
                frame = nepi_img.adjust_auto(frame,0.3)
            ##  Need to get current framerate setting
            current_fps = self.current_fps
            ##  Hard Coded for now
            framerate_ratio = self.current_controls.get("framerate_mode")/3
            [frame,new_rate] = nepi_img.adjust_framerate(frame, current_fps, framerate_ratio)
        return frame

    def setDriverCameraControl(self, control_name, value):
        # Don't log too fast -- slider bars, etc. can cause this to get called many times in a row
        #rospy.loginfo_throttle(1.0, self.node_name + ": updating driver camera control " + control_name)
        return self.driver.setScaledCameraControl(control_name, value)
    
    # Good base class candidate - Shared with ONVIF
    def getColorImg(self):
        self.img_lock.acquire()
        # Always try to start image acquisition -- no big deal if it was already started; driver returns quickly
        ret, msg = self.driver.startImageAcquisition()
        if ret is False:
            self.img_lock.release()
            return ret, msg, None, None
        
        self.color_image_acquisition_running = True

        timestamp = None
        
        start = time.time()
        frame, timestamp, ret, msg = self.driver.getImage()
        stop = time.time()
        #print('GI: ', stop - start)
        if ret is False:
            self.img_lock.release()
            return ret, msg, None, None
        
        if timestamp is not None:
            ros_timestamp = rospy.Time.from_sec(timestamp)
        else:
            ros_timestamp = rospy.Time.now()
        
        # Apply controls
        frame = self.applyControls2Frame(frame)

        # Make a copy for the bw thread to use rather than grabbing a new frame
        if self.bw_image_acquisition_running:
            self.cached_2d_color_frame = frame
            self.cached_2d_color_frame_timestamp = ros_timestamp



        self.img_lock.release()
        return ret, msg, frame, ros_timestamp
    
    # Good base class candidate - Shared with ONVIF
    def stopColorImg(self):
        self.img_lock.acquire()
        # Don't stop acquisition if the b/w image is still being requested
        if self.bw_image_acquisition_running is False:
            ret,msg = self.driver.stopImageAcquisition()
        else:
            ret = True
            msg = "Success"
        self.color_image_acquisition_running = False
        self.cached_2d_color_frame = None
        self.cached_2d_color_frame_timestamp = None
        self.img_lock.release()
        return ret,msg
    
    # Good base class candidate - Shared with ONVIF
    def getBWImg(self):
        self.img_lock.acquire()
        # Always try to start image acquisition -- no big deal if it was already started; driver returns quickly
        ret, msg = self.driver.startImageAcquisition()
        if ret is False:
            self.img_lock.release()
            return ret, msg, None, None
        
        self.bw_image_acquisition_running = True

        ros_timestamp = None
        
        # Only grab a frame if we don't already have a cached color frame... avoids cutting the update rate in half when
        # both image streams are running
        if self.color_image_acquisition_running is False or self.cached_2d_color_frame is None:
            #rospy.logwarn("Debugging: getBWImg acquiring")
            frame, timestamp, ret, msg = self.driver.getImage()
            if timestamp is not None:
                ros_timestamp = rospy.Time.from_sec(timestamp)
            else:
                ros_timestamp = rospy.Time.now()
            # Apply controls
            frame = self.applyControls2Frame(frame)
        else:
            #rospy.logwarn("Debugging: getBWImg reusing")
            frame = self.cached_2d_color_frame.copy()
            ros_timestamp = self.cached_2d_color_frame_timestamp
            self.cached_2d_color_frame = None # Clear it to avoid using it multiple times in the event that threads are running at different rates
            self.cached_2d_color_frame_timestamp = None
            ret = True
            msg = "Success: Reusing cached frame"

        self.img_lock.release()

        # Abort if there was some error or issue in acquiring the image
        if ret is False or frame is None:
            return False, msg, None, None

        # Fix the channel count if necessary
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        return ret, msg, frame, ros_timestamp
    
    # Good base class candidate - Shared with ONVIF
    def stopBWImg(self):
        self.img_lock.acquire()
        # Don't stop acquisition if the color image is still being requested
        if self.color_image_acquisition_running is False:
            ret,msg = self.driver.stopImageAcquisition()
        else:
            ret = True
            msg = "Success"
        self.bw_image_acquisition_running = False
        self.img_lock.release()
        return ret, msg

        
if __name__ == '__main__':
    node = V4l2CameraNode()
