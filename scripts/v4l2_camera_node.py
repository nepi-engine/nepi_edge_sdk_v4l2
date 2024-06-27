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

from nepi_edge_sdk_base import nepi_ros
from nepi_edge_sdk_base import nepi_img
from nepi_edge_sdk_base import nepi_idx

class V4l2CameraNode:
    DEFAULT_NODE_NAME = "v4l2_camera_node"

    FACTORY_SETTINGS_OVERRIDES = dict( white_balance_temperature_auto = "True",
                                    focus_auto = "True" )

    #Factory Control Values 
    FACTORY_CONTROLS = dict( controls_enable = True,
    auto_adjust = False,
    brightness_ratio = 0.5,
    contrast_ratio =  0.5,
    threshold_ratio =  0.0,
    resolution_mode = 3, # LOW, MED, HIGH, MAX
    framerate_mode = 3, # LOW, MED, HIGH, MAX
    start_range_ratio = None, 
    stop_range_ratio = None,
    min_range_m = None,
    max_range_m = None,
    frame_id = None
    )
 
    DEFAULT_CURRENT_FPS = 20 # Will be update later with actual

    DRIVER_SPECIALIZATION_CONSTRUCTORS = {V4L2_GENERIC_DRIVER_ID: V4l2CamDriver} # Only the generic on at the moment

    device_info_dict = dict(node_name = "",
                            sensor_name = "",
                            identifier = "",
                            serial_number = "",
                            hw_version = "",
                            sw_version = "")


    # Create threading locks, controls, and status
    img_lock = threading.Lock()
    color_image_acquisition_running = False
    bw_image_acquisition_running = False
    cached_2d_color_image = None
    cached_2d_color_image_timestamp = None

    def __init__(self):
        # Launch the ROS node
        rospy.loginfo("")
        rospy.loginfo("********************")
        rospy.loginfo("Starting " + self.DEFAULT_NODE_NAME)
        rospy.loginfo("********************")
        rospy.loginfo("")
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
                "Range":  None
            },
            

            "Data" : {
                # Data callbacks
                "Color2DImg": self.getColorImg,
                "StopColor2DImg": self.stopColorImg,
                "BW2DImg": self.getBWImg,
                "StopBW2DImg": self.stopBWImg,
                "DepthMap": None, 
                "StopDepthMap": None,
                "DepthImg": None, 
                "StopDepthImg": None,
                "Pointcloud": None, 
                "StopPointcloud": None,
                "PointcloudImg": None, 
                "StopPointcloudImg": None,
                "GPS": None,
                "Odom": None,
                "Heading": None,
            }
        }

        # Initialize controls
        self.factory_controls = self.FACTORY_CONTROLS
        self.current_controls = self.factory_controls # Updateded during initialization
        self.current_fps = self.DEFAULT_CURRENT_FPS # Should be updateded when settings read


        # Initialize settings
        self.cap_settings = self.getCapSettings()
        rospy.loginfo("CAPS SETTINGS")
        #for setting in self.cap_settings:
            #rospy.loginfo(setting)
        self.factory_settings = self.getFactorySettings()
        rospy.loginfo("FACTORY SETTINGS")
        #for setting in self.factory_settings:
            #rospy.loginfo(setting)


        # Launch the IDX interface --  this takes care of initializing all the camera settings from config. file
        rospy.loginfo(self.node_name + ": Launching NEPI IDX (ROS) interface...")
        self.device_info_dict["node_name"] = self.node_name
        if self.node_name.find("_") != -1:
            split_name = self.node_name.rsplit('_', 1)
            self.device_info_dict["sensor_name"] = split_name[0]
            self.device_info_dict["identifier"] = split_name[1]
        else:
            self.device_info_dict["sensor_name"] = self.node_name
        self.idx_if = ROSIDXSensorIF(device_info = self.device_info_dict,
                                     capSettings = self.cap_settings,
                                     factorySettings = self.factory_settings,
                                     settingUpdateFunction=self.settingUpdateFunction,
                                     getSettingsFunction=self.getSettings,
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
    # Sensor setting functions

    def getCapSettings(self):
        settings = []
        controls_dict = self.driver.getCameraControls()
        for setting_name in controls_dict.keys():
            info = controls_dict[setting_name]
            setting_type = info['type']
            if setting_type == 'int':
                setting_type = 'Int'
            elif setting_type == 'float':
                setting_type = 'Float'
            elif setting_type == 'bool':
                setting_type = 'Bool'
            elif setting_type == 'menu':
                setting_type = 'Menu'
            setting = [setting_type,setting_name]
            if setting_type == 'Float' or setting_type == 'Int':
                setting_min = str(info['min'])
                setting_max = str(info['max'])
                setting.append(setting_min)
                setting.append(setting_max)
            elif setting_type == 'Menu':
                legend = info['legend']
                for option_name in legend.keys():
                    option_ind = legend[option_name]
                    setting.append(option_name + ":" + str(option_ind))
            settings.append(setting)
        [success,available_resolutions] = self.driver.getCurrentFormatAvailableResolutions()
        setting_type = 'Discrete'
        setting_name = 'resolution'
        setting=[setting_type,setting_name]
        for res_dict in available_resolutions:
            width = str(res_dict['width'])
            height = str(res_dict['height'])
            setting_option = (width + ":" + height)
            setting.append(setting_option)
        settings.append(setting)
        return settings

    def getFactorySettings(self):
        settings = []
        controls_dict = self.driver.getCameraControls()
        for setting_name in controls_dict.keys():
            info = controls_dict[setting_name]
            setting_type = info['type']
            if setting_type == 'int':
                setting_type = 'Int'
            elif setting_type == 'float':
                setting_type = 'Float'
            elif setting_type == 'bool':
                setting_type = 'Bool'
            elif setting_type == 'menu':
                setting_type = 'Menu'
            # Create Menu Setting
            if setting_type == 'Menu':
                setting_default = 'None'
                setting_value = info['default']
                legend = info['legend']
                for menu_name in legend.keys():
                    menu_value = legend[menu_name]
                    if menu_value == setting_value:
                        setting_default = (menu_name + ":" + str(setting_value))
            else:
                setting_default = str(info['default'])
            setting = [setting_type,setting_name,setting_default]
            settings.append(setting)
        [success,res_dict] = self.driver.getCurrentResolution()
        setting_type = 'Discrete'
        setting_name = 'resolution'
        setting=[setting_type,setting_name]
        width = str(res_dict['width'])
        height = str(res_dict['height'])
        setting_option = (width + ":" + height)
        setting.append(setting_option)
        settings.append(setting)
        #Apply factory setting overides
        for setting in settings:
            if setting[1] in self.FACTORY_SETTINGS_OVERRIDES:
                setting[2] = self.FACTORY_SETTINGS_OVERRIDES[setting[1]]
                settings = nepi_ros.update_setting_in_settings(setting,settings)
        return settings
            

    def getSettings(self):
        settings = []
        controls_dict = self.driver.getCameraControls()
        #for key in controls_dict.keys():
            #string = str(controls_dict[key])
            #rospy.loginfo(key + " " + string)
        for setting_name in controls_dict.keys():
            info = controls_dict[setting_name]
            setting_type = info['type']
            if setting_type == 'int':
                setting_type = 'Int'
            elif setting_type == 'float':
                setting_type = 'Float'
            elif setting_type == 'bool':
                setting_type = 'Bool'
            elif setting_type == 'menu':
                setting_type = 'Menu'
            # Create Current Setting
            if setting_type == 'Menu':
                setting_current = 'None'
                setting_value = info['value']
                legend = info['legend']
                for menu_name in legend.keys():
                    menu_value = legend[menu_name]
                    if menu_value == setting_value:
                        setting_current = (menu_name + ":" + str(setting_value))
            else:
                setting_current = str(info['value'])
            setting = [setting_type,setting_name,setting_current]
            settings.append(setting)
        [success,res_dict] = self.driver.getCurrentResolution()
        setting_type = 'Discrete'
        setting_name = 'resolution'
        setting=[setting_type,setting_name]
        width = str(res_dict['width'])
        height = str(res_dict['height'])
        setting_option = (width + ":" + height)
        setting.append(setting_option)
        settings.append(setting)
        return settings

    def settingUpdateFunction(self,setting):
        success = False
        setting_str = str(setting)
        if len(setting) == 3:
            setting_type = setting[0]
            setting_name = setting[1]
            [s_name, s_type, data] = nepi_ros.get_data_from_setting(setting)
            if data is not None:
                setting_data = data
                found_setting = False
                for cap_setting in self.cap_settings:
                    if setting_name in cap_setting:
                        found_setting = True
                        if setting_name != "resolution":
                            success, msg = self.driver.setCameraControl(setting_name,setting_data)
                            if success:
                                msg = ( self.node_name  + " UPDATED SETTINGS " + setting_str)
                        else:
                            if data.find("(") == -1 and data.find(")") == -1: # Make sure not a function call
                                data = data.split(":")
                                width = int(eval(data[0]))
                                height = int(eval(data[1]))
                                res_dict = {'width': width, 'height': height}
                                success, msg = self.driver.setResolution(res_dict)
                            else:
                                msg = (self.node_name  + " Setting value" + setting_str + " contained () chars") 
                            break     
                if found_setting is False:
                    msg = (self.node_name  + " Setting name" + setting_str + " is not supported")                 
            else:
                msg = (self.node_name  + " Setting data" + setting_str + " is None")
        else:
            msg = (self.node_name  + " Setting " + setting_str + " not correct length")
        return success, msg

    #**********************
    # IDX driver functions

    def logDeviceInfo(self):
        device_info_str = self.node_name + " info:\n"
        device_info_str += "\tDevice Path: " + self.driver.device_path + "\n"

        controls_dict = self.driver.getCameraControls()
        for key in controls_dict.keys():
            string = str(controls_dict[key])
            rospy.loginfo(key + " " + string)
        



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
        
        rospy.loginfo(device_info_str)

    
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

    def setDriverCameraControl(self, control_name, value):
        # Don't log too fast -- slider bars, etc. can cause this to get called many times in a row
        #rospy.loginfo_throttle(1.0, self.node_name + ": updating driver camera control " + control_name)
        return self.driver.setScaledCameraControl(control_name, value)
    
    # Good base class candidate - Shared with ONVIF
    def getColorImg(self):
        encoding = "bgr8"
        self.img_lock.acquire()
        # Always try to start image acquisition -- no big deal if it was already started; driver returns quickly
        ret, msg = self.driver.startImageAcquisition()
        if ret is False:
            self.img_lock.release()
            return ret, msg, None, None, None
        
        self.color_image_acquisition_running = True

        timestamp = None
        
        start = time.time()
        cv2_img, timestamp, ret, msg = self.driver.getImage()
        stop = time.time()
        #print('GI: ', stop - start)
        if ret is False:
            self.img_lock.release()
            return ret, msg, None, None, None
        
        if timestamp is not None:
            ros_timestamp = rospy.Time.from_sec(timestamp)
        else:
            ros_timestamp = rospy.Time.now()
        
        # Apply controls
        if self.current_controls.get("controls_enable") and cv2_img is not None:
          cv2_img = nepi_idx.applyIDXControls2Image(cv2_img,self.current_controls,self.current_fps)

        # Make a copy for the bw thread to use rather than grabbing a new image
        if self.bw_image_acquisition_running:
            self.cached_2d_color_image = cv2_img
            self.cached_2d_color_image_timestamp = ros_timestamp

        self.img_lock.release()
        return ret, msg, cv2_img, ros_timestamp, encoding
    
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
        self.cached_2d_color_image = None
        self.cached_2d_color_image_timestamp = None
        self.img_lock.release()
        return ret,msg
    
    # Good base class candidate - Shared with ONVIF
    def getBWImg(self):
        encoding = "mono8"
        self.img_lock.acquire()
        # Always try to start image acquisition -- no big deal if it was already started; driver returns quickly
        ret, msg = self.driver.startImageAcquisition()
        if ret is False:
            self.img_lock.release()
            return ret, msg, None, None, None
        
        self.bw_image_acquisition_running = True

        ros_timestamp = None
        
        # Only grab a image if we don't already have a cached color image... avoids cutting the update rate in half when
        # both image streams are running
        if self.color_image_acquisition_running is False or self.cached_2d_color_image is None:
            #rospy.logwarn("Debugging: getBWImg acquiring")
            cv2_img, timestamp, ret, msg = self.driver.getImage()
            if timestamp is not None:
                ros_timestamp = rospy.Time.from_sec(timestamp)
            else:
                ros_timestamp = rospy.Time.now()
            # Apply controls
            if self.current_controls.get("controls_enable") and cv2_img is not None:
                cv2_img = nepi_idx.applyIDXControls2Image(cv2_img,self.current_controls,self.current_fps)
        else:
            #rospy.logwarn("Debugging: getBWImg reusing")
            cv2_img = self.cached_2d_color_image.copy()
            ros_timestamp = self.cached_2d_color_image_timestamp
            self.cached_2d_color_image = None # Clear it to avoid using it multiple times in the event that threads are running at different rates
            self.cached_2d_color_image_timestamp = None
            ret = True
            msg = "Success: Reusing cached cv2_img"

        self.img_lock.release()

        # Abort if there was some error or issue in acquiring the image
        if ret is False or cv2_img is None:
            return False, msg, None, None, None

        # Fix the channel count if necessary
        if cv2_img.ndim == 3:
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        
        return ret, msg, cv2_img, ros_timestamp, encoding
    
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
