#!/usr/bin/env python
#
# Copyright (c) 2024 Numurus, LLC <https://www.numurus.com>.
#
# This file is part of nepi-engine
# (see https://github.com/nepi-engine).
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
#
import sys, subprocess
import cv2
import threading, time

# Leverages OpenCV and v4l2-ctl utility to query and control camera
# TODO: Would be nicer to leverage a module rather than v4l2-ctl as
# control and status interface. This one may be nice once all is
# ported to Python 3: https://pypi.org/project/v4l2py/

V4L2_GENERIC_DRIVER_ID = 'GenericV4l2'

class V4l2CamDriver(object):
  MAX_CONSEC_FRAME_FAIL_COUNT = 3
  
  def __init__(self, device_path):
    self.device_path = device_path
    self.v4l2ctl_prefix = ['v4l2-ctl', '-d', str(self.device_path)]

    # First check that the desired device exists:
    p = subprocess.Popen(self.v4l2ctl_prefix + ['--list-devices'],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,
                           text=True)
    stdout,_ = p.communicate()
    if p.returncode != 0:
      raise Exception("Failed to list v4l2 devices: " + stdout)
    
    out = stdout.splitlines()

    path_validated = False
    tmp_device_type = None
    nLines = len(out)
    for i in range(0, nLines):
      line = out[i].strip()
      if line.endswith(':'): # Start of a new block of devices
        tmp_device_type = line.split()[0]
      elif (tmp_device_type != None) and (line == device_path):
        path_validated = True # More verification to come
        self.device_type = tmp_device_type
        break

    if not path_validated:
      raise Exception("Failed to find a camera at " + device_path)

    self.connected = True
    status, msg = self.initCameraControlsDict()
    if status is False:
      self.connected = False
      raise Exception("Failed to identify camera controls: " + msg)
    status, msg = self.initVideoFormatDict()
    if status is False:
      self.connected = False
      raise Exception("Failed to identify video formats: " + msg)
    
    self.img_acq_lock = threading.Lock()
    self.v4l2_cap = None # Delayed until streaming begins
    self.latest_frame = None
    self.latest_frame_timestamp = None
    self.latest_frame_success = False
    self.img_acq_thread = None
    self.consec_failed_frames = 0

  def isConnected(self):
    return self.connected

  def initCameraControlsDict(self):
    # Adapted from GitHub user 3DSF https://gist.github.com/3dsf/62dbe5c3636276289a719da246f6d95c

    p = subprocess.Popen(self.v4l2ctl_prefix + ['--list-ctrls-menu'],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,
                           text=True)
    stdout,_ = p.communicate()
    if p.returncode != 0:
      return False, "Failed to query controls from v4l2 device"
    
    out = stdout.splitlines()

    self.camera_controls = dict()

    nLines = len(out)
    for i in range(0, nLines):
      #Skip menu legend lines which are denoted by 4 tabs
      if out[i].startswith('\t\t\t\t'):
        continue

      a = dict()
      setting = out[i].split(':',1)[0].split()   
              # ['brightness', '0x00980900', '(int)']
      param = out[i].split(':',1)[1].split()     
              # ['min=-64', 'max=64', 'step=1', 'default=0', 'value=0']

      # Validate that this is a real settings list and skip it if not
      if (len(setting) != 3):
        continue
      try:
        a['bitName'] = int(setting[1], base=0)
      except:
        continue
      setting_type = setting[2].strip("()")
      if (not setting_type == "menu") and (not setting_type == "int") and (not setting_type == "bool"):
        continue
      
      a['type'] = setting_type      

      # Put parameters into a dictionary
      for j in range(0, len(param)):
        key,value = param[j].split('=',2)
        if (a['type'] == 'int' or a['type'] == 'menu'):
          try:
            a[key] = int(value)
          except:
            a[key] = value
        elif (a['type'] == 'bool'):
          try:
            int_val = int(value)
            a[key] = True if value == 1 else False
          except:
            a[key] = value
        
        # Build a legend for a discrete menu
        if (a['type'] == 'menu'):
          h = 0
          legend = dict()
          while h >= 0:
            h += 1
            ih = i + h
            if out[ih].startswith('\t\t\t\t') and (ih) <= nLines:
              legend_value,legend_key = out[i+h].strip().split(':')
              legend[legend_key.strip()] = int(legend_value)
            else:
              h = -1
          a['legend'] = legend    # additional data on settings
        a['step'] = 1           # adding to work with updateUVCsetting()
        
      # Use setting name as key and dictionary of params as value
      self.camera_controls.update({setting[0]: a})

    # Debugging
    #for x in self.camera_controls:
    #  print(x)
    #  print('\t' + str(self.camera_controls[x]))
    return True, "Success"

  def initVideoFormatDict(self):
    p = subprocess.Popen(self.v4l2ctl_prefix + ['--list-formats-ext'],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,
                           text=True)
    stdout,_ = p.communicate()
    if p.returncode != 0:
      return False, "Failed to query video formats from v4l2 device"
    
    out = stdout.splitlines()

    self.video_formats = list()
    tmp_format = dict()
    tmp_resolutions = dict()
    in_video_capture_format = False

    nLines = len(out)
    for i in range(0, nLines):
      if (not out[i]) or out[i].isspace():
        continue
      line = out[i].strip()
      #print("Debugging: Processing " + line)
      key, value = line.split(':', 1)
      key = key.strip()
      value = value.strip()
      #Skip unimportant lines
      if key == 'ioctl':
        continue

      #print("Debugging: Processing " + key + ", " + value)
      if key == 'Type':
        if value == 'Video Capture':
          in_video_capture_format = True
        else:
          in_video_capture_format = False

      elif key.startswith('['):
        # Add previous dictionary and start a new one
        if tmp_format:
          if tmp_resolutions: # Close out last resolution from previous index
            tmp_format['resolutions'].append(tmp_resolutions)
          self.video_formats.append(tmp_format)
          tmp_format = dict()
        tmp_format['format'] = value.split()[0].strip('\'')
        tmp_format['resolutions'] = list()      
      
      elif key == 'Size' and in_video_capture_format:
        if tmp_resolutions:
          # Add previous dictionary and start a new one
          tmp_format['resolutions'].append(tmp_resolutions)
          tmp_resolutions = dict()
        resolution_width, resolution_height = value.split()[1].split('x')
        tmp_resolutions = {'width' : int(resolution_width), 'height' : int(resolution_height)}
        tmp_resolutions['framerates'] = list()
      elif key == 'Interval' and in_video_capture_format:
        fps_val = value.split()[2].strip('()').split()[0]
        tmp_resolutions['framerates'].append(float(fps_val))

    if in_video_capture_format is True:
      if tmp_resolutions:
        tmp_format['resolutions'].append(tmp_resolutions)
      self.video_formats.append(tmp_format)

    #print("Debugging (Video Formats): " + str(self.video_formats))
    return True, "Success"


  def getCameraControls(self):
    return self.camera_controls

  def setCameraControl(self, setting_name, val):
    if not setting_name in self.camera_controls:
      return False, "Unavailable setting: " + setting_name
    if self.camera_controls[setting_name]['type'] == 'bool':
      new_val = int(val)
    else:
      new_val = val

    p = subprocess.Popen(self.v4l2ctl_prefix + ['--set-ctrl', setting_name + '=' + str(new_val)],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,
                           text=True)
    stdout,_ = p.communicate()
    if p.returncode != 0:
      return False, "Failed to set camera control to v4l2 device"
    if stdout:
      return False, "v4l2-ctl failed: " + stdout
    [val_check,msg] = self.getCameraControl(setting_name)
    if val_check != new_val:
      return False, ( "Control did not update from " + str(val_check) + " to " + str(val) + " with msg " + msg ) 
    self.camera_controls[setting_name]['value'] = val # Update controls dictionary
    return True, "Success"

  def getCameraControl(self, setting_name):
    if not setting_name in self.camera_controls:
      return False, "Unavailable setting: " + setting_name

    p = subprocess.Popen(self.v4l2ctl_prefix + ['--get-ctrl', setting_name],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,
                           text=True)
    stdout,_ = p.communicate()
    if p.returncode != 0:
      return -1.0, "Failed to get camera control value from v4l2 device"

    val = 0
    try:
      string_val = stdout.split(':')[1]
      val = int(string_val)
    except:
      return -1.0, "Failed to convert camera control to numeric value"
    if self.camera_controls[setting_name]['type'] == 'bool':
      val = bool(val)
    return val, "Success"

  def hasAdjustableCameraControl(self, setting_name):
    if setting_name not in self.camera_controls:
      return False
    
    return True
  
  def hasAdjustableResolution(self):
    return (len(self.getCurrentFormatAvailableResolutions()[1]) > 1)
  
  def hasAdjustableFramerate(self):
    return (len(self.getCurrentResolutionAvailableFramerates()[1]) > 1)    

  def getCurrentVideoSettings(self):
    p = subprocess.Popen(self.v4l2ctl_prefix + ['--get-fmt-video'],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,
                           text=True)
    stdout,_ = p.communicate()
    if p.returncode != 0:
      return False, {}
    
    out = stdout.splitlines()

    video_settings_dict = dict()
    nLines = len(out)
    for i in range(0, nLines):
      line = out[i].strip()
      tokens = line.split(':')
      if (len(tokens) < 2) or (len(tokens[0]) == 0) or (len(tokens[1]) == 0):
        continue
      key = tokens[0].strip()
      value = tokens[1].split()[0]
      
      if key == 'Width/Height':
        width,height = value.split('/')
        video_settings_dict['width'] = int(width)
        video_settings_dict['height'] = int(height)
      elif key == 'Pixel Format':
        video_settings_dict['format'] = value.strip('\'')

    return True, video_settings_dict
  
  def getCurrentResolution(self):
    status, video_settings_dict = self.getCurrentVideoSettings()
    if status is False:
      return status, {"width":0, "height":0}
    
    return True, {"width":video_settings_dict["width"], "height":video_settings_dict["height"]}
  
  def setResolution(self, resolution_dict):
    # First verify that the selected resolution is allowed
    status, available_resolutions = self.getCurrentFormatAvailableResolutions()
    if status is False:
      return False, "Failed to query current available resolutions"
    
    resolution_is_valid = False
    for available_res in available_resolutions:
      if (available_res["width"] == resolution_dict["width"]) and (available_res["height"] == resolution_dict["height"]):
        resolution_is_valid = True
        break

    if not resolution_is_valid:
      return False, "Requested resolution is not available"
    
    status, curr_resolution_dict = self.getCurrentResolution()
    if status is False:
      return False, "Failed to obtain current resolution"
    
    # Don't need to do anything if resolution is already as desired
    if (resolution_dict["width"] == curr_resolution_dict["width"]) and (resolution_dict["height"] == curr_resolution_dict["height"]):
      return True, "Desired resolution already set"

    status, format = self.getCurrentFormat()
    if status is False:
      return False, "Failed to obtain current format"
    
    # Must stop image acquisition while resolution gets updated
    lock_held = False
    if self.imageAcquisitionRunning() is True:
      self.stopImageAcquisition(hold_lock=True) # Parent must restart
      lock_held = True

    resolution_cmd = "--set-fmt-video=width=" + str(resolution_dict["width"]) + ",height=" + str(resolution_dict["height"]) + ",pixelformat=" + format
    p = subprocess.Popen(self.v4l2ctl_prefix + [resolution_cmd],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         text=True)
    stdout,_ = p.communicate()
    if p.returncode != 0:
      if lock_held:
        self.img_acq_lock.release()
      return False, "Failed to set resolution for v4l2 device (" + stdout + ")"
      
    if lock_held is True:
      self.img_acq_lock.release()
        
    return True, "Success"
  
  def getCurrentResolutionAvailableFramerates(self):
    status, curr_video_settings = self.getCurrentVideoSettings()
    if status is False:
      return False, "Failed to get current video settings for framerate query"
    
    framerates = list()
    
    for entry in self.video_formats:
      if entry['format'] != curr_video_settings['format']:
        continue
      for resolution in entry['resolutions']:
        if (resolution["width"] == curr_video_settings["width"]) and (resolution["height"] == curr_video_settings["height"]):
          framerates = list(resolution["framerates"])
          framerates.sort()
          return True, framerates
        
    return False, "Failed to identify framerates for current resolution"

  def setFramerate(self, max_fps):
    status, available_framerates = self.getCurrentResolutionAvailableFramerates()
    if status is False:
      return False, "Failed to query current available framerates"
    
    framerate_valid = False
    for r in available_framerates:
      if max_fps == r:
        framerate_valid = True
        break

    if not framerate_valid:
      return False, "Invalid framerate requested"
    
    status, curr_fps = self.getFramerate()
    if status is False:
      return False, "Unable to check current framerate during update"
    
    if curr_fps == max_fps:
      return True, "Current framerate is already as desired"
    
    lock_held = False
    if self.imageAcquisitionRunning() is True:
      #needs_img_acq_restart = True
      self.stopImageAcquisition(hold_lock=True) # Parent must restart
      lock_held = True

    p = subprocess.Popen(self.v4l2ctl_prefix + ['--set-parm=' + str(max_fps)],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         text=True)
    stdout,_ = p.communicate()
    if p.returncode != 0:
      if lock_held:
        self.img_acq_lock.release()
      return False, "Failed to set framerate for v4l2 device: " + stdout

    if lock_held:
      self.img_acq_lock.release()

    return True, ""

  def getFramerate(self):
    p = subprocess.Popen(self.v4l2ctl_prefix + ['--get-parm'],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         text=True)
    stdout,_ = p.communicate()
    if p.returncode != 0:
      return False, "Failed to get framerate for v4l2 device: " + stdout

    out = stdout.splitlines()
    nLines = len(out)
    for i in range(0, nLines):
      line = out[i].strip()
      key, value = line.split(':')
      value = value.strip()
      if key == 'Frames per second':
        fps = float(value.split()[0])
        return True, fps
    
    return False, "Failed to obtain current framerate"

  def getCurrentFormat(self):
    status, video_settings_dict = self.getCurrentVideoSettings()
    if status is False:
      return False, "Unknown"
    
    return True, video_settings_dict["format"]
  
  def getCurrentFormatAvailableResolutions(self):
    status, format = self.getCurrentFormat()
    if status is False:
      return status, []

    resolution_list = list()
    for entry in self.video_formats:
        if entry["format"] == format:
          for resolution in entry["resolutions"]:
            resolution_list.append({"width": resolution["width"], "height": resolution["height"]})

    resolution_list_sorted = sorted(resolution_list, key=lambda x: x["width"])
    #print("Debugging: " + str(resolution_list_sorted))
    return True, resolution_list_sorted
  
  def imageAcquisitionRunning(self):
    return (self.v4l2_cap != None)
  
  def startImageAcquisition(self):
    self.img_acq_lock.acquire()
    if self.v4l2_cap != None:
      self.img_acq_lock.release()
      return True, "Already capturing from v4l2 device " + self.device_path
    
    # Get the video settings so that we can (re)apply them to the capture device -- otherwise OpenCV overwrites these
    status, curr_video_settings = self.getCurrentVideoSettings()
    if status is False:
      self.img_acq_lock.release()
      return False, "Failed to get current video settings prior to starting image acquisition"
    
    status, curr_frame_rate = self.getFramerate()
    if status is False:
      self.img_acq_lock.release()
      return False, "Failed to obtain framerate before starting image acquisition"

    # Create the OpenCV cap object using V4L2 as the backend API -- TODO: Maybe other APIs would be better? (Seems FFMPEG doesn't work for USB cams, however)
    self.v4l2_cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)

    if not self.v4l2_cap.isOpened():
      self.img_acq_lock.release()
      return False, "Failed to open capture object for v4l2 device " + self.device_path 

    # Experimental: Try setting a small buffer size for reduced latency
    self.v4l2_cap.set(cv2.CAP_PROP_BUFFERSIZE, 3) # Not sure this does anything

    # Set the relevant video settings to the cv2 capture device to ensure they are as desired
    self.v4l2_cap.set(cv2.CAP_PROP_FRAME_WIDTH, curr_video_settings["width"])
    self.v4l2_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, curr_video_settings["height"])
    self.v4l2_cap.set(cv2.CAP_PROP_FPS, curr_frame_rate)
    
    self.latest_frame = None
    self.latest_frame_timestamp = None
    self.latest_frame_success = False
    self.img_acq_lock.release()
    # Experimental: Launch a separate thread to grab frames as quickly as possible so that the buffer is empty when downstream
    # client actually wants a real image
    self.img_acq_thread = threading.Thread(target=self.runImgAcqThread)
    self.img_acq_thread.daemon = True
    self.img_acq_thread.start()
    
    self.consec_failed_frames = 0

    return True, "Success"
  
  def runImgAcqThread(self):
    if self.v4l2_cap is None or self.v4l2_cap.isOpened() is False:
       return
    
    keep_going = True
    while(keep_going):
      self.img_acq_lock.acquire()
      if self.v4l2_cap is None:
        keep_going = False
      else:
        # Acquire without decoding via grab(). Image is available via a subsequent retrieve()
        start = time.time()
        self.latest_frame_success = self.v4l2_cap.grab()
        self.latest_frame_timestamp = time.time()
        stop = self.latest_frame_timestamp
        #print('G: ', stop - start)
                          
      self.img_acq_lock.release()
      time.sleep(0.01)

  def stopImageAcquisition(self, hold_lock=False):
    self.img_acq_lock.acquire()
    if self.v4l2_cap == None:
        if not hold_lock:
          self.img_acq_lock.release()
        return True, "No current capture from " + self.device_path
    
    # Experimental: Try clearing the queue
    for i in range(10):
      self.v4l2_cap.grab()

    self.v4l2_cap.release()
    self.v4l2_cap = None
    
    if not hold_lock:
      self.img_acq_lock.release()
    return True, "Success"
  
  def getImage(self):
    self.img_acq_lock.acquire()
    if self.v4l2_cap is None or self.v4l2_cap.isOpened() is False:
        self.img_acq_lock.release()
        return None, None, False, "Capture for " + self.device_path + " not opened"
    
    # Just decode and return latest grabbed by acquisition thread
    if self.latest_frame_success is True:
        start = time.time()
        ret, self.latest_frame = self.v4l2_cap.retrieve()
        stop = time.time()
        #print('R: ', stop - start)
    else:
        ret = False
        self.latest_frame = None
    
    self.img_acq_lock.release()
    if not ret:
        self.consec_failed_frames += 1
        if self.consec_failed_frames < self.MAX_CONSEC_FRAME_FAIL_COUNT:
            return None, None, False, "Failed to read next frame for " + self.device_path
        else:
            self.stopImageAcquisition()
            self.startImageAcquisition()
            return None, None, False, "Failed to read next frame " + str(self.MAX_CONSEC_FRAME_FAIL_COUNT) + "times consec... auto-restarting image acquisition"
    
    return self.latest_frame, self.latest_frame_timestamp, True, "Success"

if __name__ == '__main__':
  path = '/dev/video0' if len(sys.argv) == 1 else sys.argv[1]
  driver = V4l2CamDriver(device_path=path)

  if not driver.isConnected():
    print("Failed to connect to v4l2 device " + id)
    sys.exit(-1)

  print("Connected to " + driver.device_type + " at " + driver.device_path)

  print("Supported camera controls:")
  print("\tScaled:" + str(driver.getAvailableScaledCameraControls()))
  print("\tDiscrete:" + str(driver.getAvailableDiscreteCameraControls()))

  status, current_format = driver.getCurrentFormat()
  status, current_resolution_dict = driver.getCurrentResolution()
  print("Current video settings: Format = " + str(current_format) + ", Resolution = " + str(current_resolution_dict))

  status, resolutions_list = driver.getCurrentFormatAvailableResolutions()
  print("Available resolutions for current format:")
  print("\t" + str(resolutions_list))

  status, framerates_list = driver.getCurrentResolutionAvailableFramerates()
  print("Available framerates for current resolution:")
  print("\t" + str(framerates_list))
  
  if (driver.hasAdjustableCameraControl('brightness')):
    driver.setScaledCameraControl('brightness', 0.5)
    print("New Brightness:" + str(driver.getScaledCameraControl('brightness')))
  else:
    print("No brightness setting")

  status, msg = driver.startImageAcquisition()
  if status is False:
    print("Failed to start image acquisition")
    sys.exit(-1)

  # Set up a window
  window_title = "L4V2 Driver Test"
  window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

  while(True):
    frame, timestamp, status, msg = driver.getImage()
    
    if status is True:
      if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
          cv2.imshow(window_title, frame)
      else:
          break
      keyCode = cv2.waitKey(10) & 0xFF
      # Stop the program on the ESC key or 'q'
      if keyCode == 27 or keyCode == ord('q'):
          break
      elif keyCode == ord('0'):
        driver.setResolution(resolutions_list[0])
      elif keyCode == ord('1'):
        driver.setResolution(resolutions_list[1])
      elif keyCode == ord('2'):
        driver.setResolution(resolutions_list[2])
      elif keyCode == ord('f'):
        status, framerates_list = driver.getCurrentResolutionAvailableFramerates()
        driver.setFramerate(framerates_list[0])
      elif keyCode == ord('F'):
        status, framerates_list = driver.getCurrentResolutionAvailableFramerates()
        driver.setFramerate(framerates_list[1])
    else:
      print("Failed to get image")
    
    time.sleep(0.01)
