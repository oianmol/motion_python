# MotionSMC

## Introduction

MotionSMC was created to supersede the motion-project project which was written in C and hard to maintain. Motion SMC is written in Python and uses open CV under the hood.
Since open CV is a computer vision library, it allows us to detect motion in camera videos efficiently and more
accurately.
We use a combination or different computer vision algorithms like background-subtraction and gaussian blur.

## How to use motion SMC?

Motion SMC supports passing configuration file as program argument. If not provided then it would look for an *.ini file in the user's home directory.

``$ python main.py -c motion_smc.ini``

Executing the above command, will start the motion SMC project.

## Config File Explained

### Default Config
```Ini
[DEFAULT]
regions=0 0 640 0 640 320 0 320
blur=5
post_motion_wait=3
method=MOG2
cameras=83
detect_shadows=true
output_motion_video=false
```

> **regions** Used to specify the region of interest

> **blur** the amount of blur used to remove noise

> **post_motion_wait** the number of seconds to wait before emitting motion end event

> **method** the motion algorithm to use for background subtraction

> **cameras** the total number of cameras!

> **detect_shadows** True if we want the algorithm to not consider shadows as motion.

> **output_motion_video**  used for debugging purposes, set to True if you want the motion only video as output.

### Camera Config

```Ini
[camera_0]
uri=/Users/anmolverma/testsee.mkv
area=5
camera_id=500
event_path=/Users/anmolverma/outputs/566/camera_0/
width=640
height=320
fps=4
```

> **[camera_0]** The number Zero here stands for the index of camera

> **uri**  The URI of the camera stream can be an RTP url also

> **area** The amount of area used by the algorithm to consider motion between contours.

> **camera_id** The unique ID of the camera should be given with every camera configuration

> **event_path** The path where you would like to output the motion events

> **width** The width of the camera frame

> **height** The height of the camera frame.

> **fps** The frames per second of the stream.

## Features

| Feature                                                  | Status |
|----------------------------------------------------------|--------|
| Logging to Disk                                          | ✅      |
| Motion Detection                                         | ✅      |
| Output Motion Times                                      | ✅      |
| Object Detection                                         | ❓      |
| Outputs Motion Video <br/>(Configurable - Default:False) | ✅      |

## Logging

Motion SMC Supports logging and allowing us to log any errors and debug statements while the project is running.

## Feedback and support

Please report any issues, usability improvements, or feature requests to our Github repo
<a href="https://github.com/oianmol/motion_python/issues">MotionSMC Issues</a>
(you will need to register to GitHub).


You can also always email us at [oianmol@icloud.com](mailto:oianmol@icloud.com).