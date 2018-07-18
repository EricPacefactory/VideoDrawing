# VideoDrawing
Collection of scripts used for drawing lines/zones on top of videos, images or RTSP streams


Tested on:
- Ubuntu 16.04
- Python 3.5.2

Requires:
- OpenCV (3.3.1+)
- numpy
- tkinter

OpenCV can be installed from pip, but has only been tested using a manual installation.
The pip installation seems to have unreliable video recording (not a problem) and GUI functionality (maybe a problem).
Tkinter was also installed separately from pip (sudo apt install python3-tk)

Also needs custom cv library when using the perspective drawer!
For now, this needs to be added manually to the project directory

