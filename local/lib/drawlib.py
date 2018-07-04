#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 08:22:31 2018

@author: eo
"""

import os
import cv2
import numpy as np
import datetime as dt
from collections import namedtuple

#  --------------------------------------------------------------------------------------------------------------------
#%% General functions

# .....................................................................................................................

# Function for quitting scripts in spyder IDE without crashing the kernel
def hardQuit():
    print("")    
    if any('SPYDER' in name for name in os.environ):
        raise SystemExit()  # Hard crash to stop that menacing spyder
    else:
        quit()  # Works nicely everywhere else!
        
# .....................................................................................................................
        
def setupVideoCapture(source):
    
    # OpenCV constants
    vc_width = 3
    vc_height = 4
    vc_fps = 5
    vc_fourcc = 6
    vc_framecount = 7
        
    # Set up video capture object
    videoObj = cv2.VideoCapture(source)
    if not videoObj.isOpened():
        print("")
        print("Couldn't open video object. Tried:")
        print(source)
        print("Closing...")
        print("")
        raise IOError

    videoName = source
    if "rtsp" in source.lower():
        # Try to grab the IP numbers out of the RTSP string
        splitSource = source.replace("@", ".").replace(":", ".").replace("/", ".").split(".")
        numsInSource = [int(eachEntry) for eachEntry in splitSource if eachEntry.isdigit()]
        only8bits = [str(eachNum) for eachNum in numsInSource if eachNum < 256]
        guessIPString = ".".join(only8bits[:4])
        videoName = " - ".join(["RTSP", guessIPString])
    else:
        videoName = os.path.basename(source)
    
    # Get video info
    vidWidth = int(videoObj.get(vc_width))
    vidHeight = int(videoObj.get(vc_height))
    vidFPS = videoObj.get(vc_fps)
    
    # Check that the FPS is valid (some RTSP reads are wrong)
    if not (5 < vidFPS < 61):
        print("")
        print("Error with FPS. Read as:", vidFPS)
        print("Expecting value between 5-61")
        print("Assuming: 30")
        vidFPS = 30
    
    # Try to grab video info that may not be available on RTSP
    try:
        vidFCC = int(videoObj.get(vc_fourcc)).to_bytes(4, 'little')
        totalFrames = max(-1, int(videoObj.get(vc_framecount)))
        totalRunTime = -1 if totalFrames < 0 else ((totalFrames - 1)/vidFPS)
    except Exception as e:
        print("")
        print("Couldn't get video info... RTSP stream?")
        vidFCC = b"Unknown"
        totalFrames = -1
        totalRunTime = -1
    
    # Print out video information
    currentDate = dt.datetime.now()
    print("")
    print(currentDate.strftime("%Y-%m-%d"))
    print("Video:", videoName)
    print("Dimensions:", vidWidth, "x", vidHeight, "@", vidFPS, "FPS")
    print("Codec:", vidFCC.decode("utf-8"))
    print("Total Frames:", totalFrames)
    
    # Print out different run time units, depending on the video length
    if totalRunTime > 3600:
        print("Run time:", "{:.1f} hours".format(totalRunTime/3600))
    elif totalRunTime > 60:
        print("Run time:", "{:.1f} mins".format(totalRunTime/60))
    else:
        print("Run time:", "{:.1f} seconds".format(totalRunTime))
    
    # Some final formating
    vidWH = (vidWidth, vidHeight)
    
    return videoObj, vidWH, vidFPS

# .....................................................................................................................

def saveRTSPFrame(inVideoObj):
    
    # OpenCV constant
    vc_pos_frames = 1
    
    rtspSave = input("Save an image for re-use? (y/n):\n")
    
    if rtspSave.strip().lower() == "y":
        
        (receivedFrame, inFrame) = inVideoObj.read()
        
        if receivedFrame:
            saveSource = guiSave(fileTypes=[["image", ".png"]])
            
            if saveSource is not None:
                cv2.imwrite(saveSource, inFrame)
                print("")
                print("Saved image:")
                print(saveSource)
        else:
            print("")
            print("Error getting frame for saving!")
            
        # Reset the video stream if possible
        inVideoObj.set(vc_pos_frames, 0)


# .....................................................................................................................

def loadImageResource(targetWH, debugSource=None):
    
    TypeOfSource = namedtuple("SourceType", ["image", "video", "rtsp"])
    
    # Select between the interactive loader and a hard-coded source
    if debugSource is None:
    
        # Get input source string and type
        inputSource, isImage, isVideo, isRTSP = rtspOrFileFromCommandLine()
        
    else:
        # Figure out what kind of input we have
        inputSource = debugSource        
        if "rtsp" in inputSource.lower():
            isImage, isVideo, isRTSP = False, False, True
        elif os.path.splitext(inputSource)[1].lower() in [".jpg", ".png", ".bmp"]:
            isImage, isVideo, isRTSP = True, False, False
        else:
            isImage, isVideo, isRTSP = False, True, False
    
    # Quick sanity check
    if sum([isImage, isVideo, isRTSP]) != 1:
        print("")
        print("Error with deciding if the file is an image, video or RTSP stream! Got:")
        print("Image:", isImage)
        print("Video:", isVideo)
        print("RTSP:", isRTSP)
        print("")
        raise TypeError
        
    # Store the type of input source, since it can affect certain looping functions
    sourceType = TypeOfSource(image=isImage, video=isVideo, rtsp=isRTSP)        
        
    # Some feedback
    print("")
    print("Using input:")
    print(inputSource)
    
    
    # If the input is a video, just load it
    if isVideo:
        videoObj, vidWH, vidFPS = setupVideoCapture(inputSource)
        
    
    # If input is RTSP stream, load it up and check if user wants to save an image (for easier re-use)
    if isRTSP:
        videoObj, vidWH, vidFPS = setupVideoCapture(inputSource)
        
        # Check if user wants to save a frame from the video
        saveRTSPFrame(videoObj)    
    
    # If the input is an image, load the image directly
    if isImage:
        
        # Create a dummy video vapture object so we can use an image (mostly) seemlessly
        class FakeVideoCapture:
            
            def __init__(self, inputImage):
                self._image = inputImage
            
            def read(self):
                return True, self._image.copy()
            
            def release(self):                
                # Not a real video, so don't nothing on release
                return
        
        inImage = cv2.imread(inputSource)
        vidWH = inImage.shape[1::-1]
        #vidFPS = 0
        
        videoObj = FakeVideoCapture(inImage)
        
    # Get scaling to decide how to appropriately resize the incoming frames
    frameScaling = 1 / max(np.array(vidWH) / np.array(targetWH))
    scaledWH = (int(frameScaling*vidWH[0]), int(frameScaling*vidWH[1]))
    
    return inputSource, videoObj, scaledWH, sourceType

# .....................................................................................................................

def loadFromHistory(fileSource):
    
    # Initialize default output
    outSource = None
    
    # Check for previously used video sources
    if os.path.exists(fileSource):
        with open(fileSource, 'r') as histFile:
            historyData = histFile.read()
        prevSource = historyData.splitlines()[0]
        
        if os.path.exists(prevSource) or "rtsp" in prevSource:
            print("")
            print("Found previously loaded video source:")
            print(prevSource)
            userResponse = input("Re-use source? (y/n):\n") 
            confirmReuse = (userResponse.lower().strip() != 'n')
            outSource = prevSource if confirmReuse else None
            
    return outSource

# .....................................................................................................................

def saveSourceHistory(fileSource, videoSource):
    
    # Create directory to store history file if it doesn't already exist
    sourceDir = os.path.dirname(fileSource)
    if not os.path.exists(sourceDir):
        os.makedirs(sourceDir)
        
    # Save video source string into file for re-use (using loadFromHistory() function)
    with open(fileSource, 'w') as histFile:
        histFile.write("".join([videoSource, "\n"]))
        
# .....................................................................................................................



# ---------------------------------------------------------------------------------------------------------------------
#%% GUI Functions
        
def guiLoad(searchDir=os.path.expanduser("~/Desktop"), windowTitle="Select a file", fileTypes=None, errorOut=True):
    
    import tkinter
    from tkinter import filedialog
    
    # Set general file types if none are specified
    if fileTypes is None:
        fileTypes = [["all", "*"]]
        
    # UI: Hide main window
    root = tkinter.Tk()
    root.withdraw()
    
    # Ask user to select file
    fileInSource = filedialog.askopenfilename(initialdir=searchDir, title=windowTitle, filetypes=fileTypes)
    
    # Get rid of UI elements
    root.destroy()    
    
    if len(fileInSource) < 1:
        
        # Hard crash if needed
        if errorOut:
            print("")
            print("Load cancelled!")
            print("")
            raise IOError
        else:
            return None
    
    return fileInSource

# .....................................................................................................................

def guiSave(searchDir=os.path.expanduser("~/Desktop"), windowTitle="Save file", fileTypes=None):
    
    import tkinter
    from tkinter import filedialog
    
    # Set general file types if none are specified
    if fileTypes is None:
        fileTypes = [["files", "*"]]
        
    # UI: Hide main window
    root = tkinter.Tk()
    root.withdraw()
    
    fileOutSource = filedialog.asksaveasfilename(initialdir=searchDir, title=windowTitle, filetypes=fileTypes)
    
    # Get rid of UI elements
    root.destroy()    
    
    if len(fileOutSource) < 1:
        print("")
        print("Save cancelled!")
        return None
    
    return fileOutSource

# .....................................................................................................................

def guiConfirm(confirmText, windowTitle="Confirmation"):
    
    import tkinter
    from tkinter import messagebox
    
    # UI: Hide main window
    root = tkinter.Tk()
    root.withdraw()
    
    # Get user response
    userResponse = messagebox.askyesno(windowTitle, confirmText)
    
    # Get rid of UI elements
    root.destroy()    
    
    return userResponse

# ---------------------------------------------------------------------------------------------------------------------
#%% RTSP functions
    
def getRTSP(ip, username="", password="", port=554, command=""):
    
    rtspSource = "".join(["rtsp://", username, ":", password, "@", ip, ":", str(port), "/", command])
    
    splitIP = ip.split(".")
    padIP = [eachNumber.zfill(3) for eachNumber in splitIP]
    blockIP = "".join(padIP)
    
    return rtspSource, blockIP

# .....................................................................................................................
    
def rtspFromCommandLine(errorOut=True):
        
    # Ask user for RTSP settings
    print("")
    print("****************** GET RTSP ******************")
    ipAddr = input("Enter IP address:\n")
    
    # If ip address is skipped, raise an error or exit function
    if ipAddr.strip() == "":        
        if errorOut:
            print("Bad IP!")
            print("")
            print("**********************************************")
            print("")
            raise ValueError
        else:
            return None
    
    def defaultInput(inputString, defaultValue):
        inputValue = input(inputString).strip()
        if inputValue == "":
            return defaultValue
        return inputValue

    # Get rtsp settings from user
    rtspUser = defaultInput("Enter username \t(default None):\n", "")
    rtspPass = defaultInput("Enter password \t(default None):\n", "")
    rtspPort = defaultInput("Enter port \t\t(default 554):\n", "554")
    rtspComm = defaultInput("Enter command \t(default None):\n", "")
        
    # Finish off blocking gfx
    print("")
    print("**********************************************")
    
    # Build dictionary for convenient output
    outRecord = {"username": rtspUser, 
                 "password": rtspPass,
                 "ip": ipAddr,
                 "port": rtspPort,
                 "command": rtspComm}
        
    return outRecord
        
# .....................................................................................................................

def rtspOrFileFromCommandLine():
    
    # Initialize all outputs to false
    isImage, isVideo, isRTSP = [False] * 3
    
    # Try to get RTSP info
    rtspRecord = rtspFromCommandLine(errorOut=False)
    if rtspRecord is not None:
        isRTSP = True
        inputSource, blockIP = getRTSP(**rtspRecord)
        return inputSource, isImage, isVideo, isRTSP
    
    # If RTSP skipped, try loading a file
    print("**********************************************")
    print("")
    print("Cancelling RTSP! Loading file instead.")
    try:
        inputSource = guiLoad()
    except IOError:
        hardQuit()
    
    # Figure out the file type to decide it this is a video or image
    fileName, fileExt = os.path.splitext(inputSource)    
    isImage = (fileExt.lower() in [".jpg", ".png", ".bmp"])
    isVideo = (not isImage)

    return inputSource, isImage, isVideo, isRTSP

# .....................................................................................................................
    




# ---------------------------------------------------------------------------------------------------------------------
#%% Scrap
    



