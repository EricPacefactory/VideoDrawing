#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 15:49:35 2018

@author: eo
"""

import os
import cv2
import numpy as np
from collections import namedtuple

from local.lib.drawlib import loadImageResource, loadFromHistory, saveSourceHistory

# ---------------------------------------------------------------------------------------------------------------------
#%% Define functions
    
# Convenience for closing OCV windows interactively
closeall = cv2.destroyAllWindows

# .....................................................................................................................

def drawLine(displayFrame, lineStart, lineEnd, lineColor, lineThickness, circleRadius):
    cv2.line(displayFrame, tuple(lineStart), tuple(lineEnd), lineColor, lineThickness)
    cv2.circle(displayFrame, tuple(lineStart), circleRadius, lineColor, -1)
    cv2.circle(displayFrame, tuple(lineEnd), circleRadius, lineColor, -1)
    
# .....................................................................................................................

def drawAllLines(displayFrame, inLines, lineColor, lineThickness, circleRadius):
    for eachLine in inLines:        
        drawLine(displayFrame, eachLine.start, eachLine.end, lineColor, lineThickness, circleRadius)

# .....................................................................................................................

def drawLineInProgress(displayFrame, linePoint, mousePoint, lineColor, lineThickness, circleRadius):
    
    # Only draw in-progress lines, which implies only 1 line point (i.e. the start point) be passed as an input
    if len(linePoint) == 1:
        drawLine(displayFrame, linePoint[0], mousePoint, lineColor, lineThickness, circleRadius)   

# .....................................................................................................................
        
def linePrintOut(newStart, newEnd, frameWH, borderWH, updating=False):
    
    # Calculate normalized values
    startOffset = newStart - borderWH
    endOffset = newEnd - borderWH
    normalizedStart = startOffset/(frameWH - np.array((1,1)))
    normalizedEnd = endOffset/(frameWH - np.array((1,1)))
    
    # Print out line data that works for config file
    print("")
    print("")
    print("****************************************")
    
    if updating:
        print("************* UPDATED LINE *************")
    else:
        print("*************** NEW LINE ***************")
        
    print("****************************************")
    print("")
    print("Using frame dimensions:", frameWH[0], "x", frameWH[1])
    print("")
    print("\tPixel values:")
    print("start: ", "({:.0f}, {:.0f})".format(*startOffset))
    print("end:   ", "({:.0f}, {:.0f})".format(*endOffset))
    print("")
    print("\tNormalized values:")
    print("start: ", "({:.4f}, {:.4f})".format(*normalizedStart))
    print("end:   ", "({:.4f}, {:.4f})".format(*normalizedEnd))
    
# .....................................................................................................................
    
def getUpdatedLine(inList, pointSelect, mouseXY):
    
    # Get the length of the line list for convenience
    listLength = len(inList)
    
    # Figure out which line contains the point we need to update
    lineIdx = pointSelect % listLength
    lineRef = inList[lineIdx]
    
    # Modify the start/end point of the line
    isStart = pointSelect < listLength
    
    # Get new line paramters
    newStart = mouseXY if isStart else lineRef.start
    newEnd = lineRef.end if isStart else mouseXY            
    newVec = newEnd - newStart
    
    return newStart, newEnd, newVec, lineIdx

# .....................................................................................................................
    
# ---------------------------------------------------------------------------------------------------------------------
#%% Define callback
    
def mouseCallback(event, mx, my, flags, param):
    
    # Record mouse x and y positions at all times (in case we need to draw line-in-progress)
    mxy = np.array((mx, my))
    param["mouse"] = mxy
    
    
    # ..................................................................................................................
    # Add object lines with left click
    
    if event == cv2.EVENT_LBUTTONDOWN:
        
        # Add the clicked point to the list of new points
        param["newPoints"].append(mxy)
            
        # If two new points are added, interpret them as start/end points of a new line
        if len(param["newPoints"]) >= 2:
            
            # Pull out parameter data for convenience
            newStartPoint = param["newPoints"][0]
            newEndPoint = param["newPoints"][1]
            newVec = newEndPoint - newStartPoint
            
            # Add the new line to our list
            param["lineList"].append(SimpleLine(newStartPoint, newEndPoint, newVec))
            
            # Clear the points used to create the new line so we don't re-use them
            param["newPoints"] = []
            
            # Finally, print out the info so it can be used for something!
            linePrintOut(newStartPoint, newEndPoint, param["frameWH"], param["borderWH"], updating=False)
        
        
    # .................................................................................................................
    # Clear lines with right-click
    # Only clean the line-in-progress or all lines, depending on the context
    
    if event == cv2.EVENT_RBUTTONDOWN:
        
        # Only clear line list if no points are in progress
        outString = "Line-in-progress cleared!"
        if len(param["newPoints"]) == 0:
            param["lineList"] = []
            outString = "All lines cleared!"
            
        # Clear line-in-progress points regardless
        param["newPoints"] = []
        
        # Some feedback
        print("")
        print(outString)
    
    
    # .................................................................................................................
    # Select nearest line-points on middle-down
    
    if event == cv2.EVENT_MBUTTONDOWN:
        
        # Don't do anything if a line is in-progress
        if len(param["newPoints"]) > 0:
            return
        
        # Get a list of all points
        allStartPoints = [eachLine.start for eachLine in param["lineList"]]
        allEndPoints = [eachLine.end for eachLine in param["lineList"]]
        allPoints = allStartPoints + allEndPoints
        
        # Check for the closest point
        minSqDist = 1E9
        bestMatchIdx = -1        
        for idx, eachPoint in enumerate(allPoints):
            
            # Calculate the distance between mouse and point
            distSq = np.sum(np.square(mxy - eachPoint))
            
            # Record the closest point
            if distSq < minSqDist:
                minSqDist = distSq
                bestMatchIdx = idx
        
        # Figure out if we need to change the point position
        distanceThreshold = 50**2            
        param["pointSelect"] = bestMatchIdx if minSqDist < distanceThreshold else None
        
        
    # .................................................................................................................
    # Move points on middle click & drag
    
    #print(flags)
    if flags == (mouseMoveOffset + cv2.EVENT_FLAG_MBUTTON):
        # Don't do anything if a line is in-progress
        if len(param["newPoints"]) > 0:
            return
        
        # Update the dragged points based on the mouse position
        pointSelect = param["pointSelect"]
        if pointSelect is not None:
            
            # Get the new start/end point so we can overwrite the previous line object
            newStart, newEnd, newVec, lineIdx = getUpdatedLine(param["lineList"], pointSelect, mxy)

            # Create new line object and replace the old line
            newLine = SimpleLine(start=newStart, end=newEnd, vec=newVec)
            param["lineList"][lineIdx] = newLine
    
    # .................................................................................................................
    # Release dragged lines on middle-up
    
    if event == cv2.EVENT_MBUTTONUP:
        
        # Print out the new line info if it had been dragged
        pointSelect = param["pointSelect"]
        if pointSelect is not None:           
            lineIdx = pointSelect % len(param["lineList"])
            lineRef = param["lineList"][lineIdx]
            linePrintOut(lineRef.start, lineRef.end, param["frameWH"], param["borderWH"], updating=True)
        
        # Reset the line selection
        param["pointSelect"] = None
        
    
# ---------------------------------------------------------------------------------------------------------------------
#%% Initialize variables

# OpenCV constants
vc_pos_frames = 1
mouseMoveOffset = 32 if os.uname().sysname == "Linux" else 0 # (32 for Linux, seemingly 0 for Mac, no idea for windows)

# Set display settings
targetWH = (640, 360)
frameDelay = 10

# Set image border values
solidBorder = cv2.BORDER_CONSTANT
borderColor = (20,20,20)
wBorder = 60
hBorder = 60

# Create a simple container for holding line data
SimpleLine = namedtuple("SimpleLine", ["start", "end", "vec"])

# ---------------------------------------------------------------------------------------------------------------------
#%% Get input source

# Set up debugging source (if needed)
#debugSource = "/home/eo/Desktop/PythonData/Shared/videos/dtb2.avi"
debugSource = None  # None -> disable autoloading

# Only try to load history file if debugging is disabled
historyFile = "local/conf/history.log"
if debugSource is None:
    debugSource = loadFromHistory(historyFile)

# Get the video object (could be an image), along with scaled size and type (image, video or rtsp)
videoSource, videoObj, scaledWH, sourceType = loadImageResource(targetWH, debugSource)

# Reduce the frame delay if the input is RTSP so we don't drop frames
if sourceType.rtsp:
    frameDelay = 1

# Initialize a variable used to pass data to mouse callback
cbData = {"mouse": (0, 0), 
          "newPoints": [], 
          "lineList": [], 
          "pointSelect": None,
          "frameWH": np.array(scaledWH), 
          "borderWH": np.array((wBorder, hBorder))}


# ---------------------------------------------------------------------------------------------------------------------
#%% Print out instructions

print("")
print("----------------------------------------------------------------")
print("------------------------- Instructions -------------------------")
print("----------------------------------------------------------------")
print("")
print("Left click:")
print("  - Add new points")
print("  - Line created automatically from 2 points")
print("  - Line coordinates will be printed on release")
print("  - Order of points determines line crossing directions")
print("")
print("Right click:")
print("  - Clear a line-in-progress (if currently drawing)")
print("  - Otherwise, clear all lines")
print("")
print("Middle click:")
print("  - Drag/move points")
print("  - Updated line coordinates will be printed out on release")
print("")
print("Exiting:")
print("  - Use q, Esc or spacebar to close the window")
print("")
print("----------------------------------------------------------------")
print("----------------------------------------------------------------")


# ---------------------------------------------------------------------------------------------------------------------
#%% Run video

# Set up window
setupWindow = "Draw Lines"
cv2.namedWindow(setupWindow)
cv2.setMouseCallback(setupWindow, mouseCallback, cbData)

while True:
    
    
    # .................................................................................................................
    # Get initial frame
    
    (receivedFrame, inFrame) = videoObj.read()        
        
    # Warning if the frame is missed
    if not receivedFrame:
        
        # Close the RTSP connection is we start losing frames
        if sourceType.rtsp:        
            print("")
            print("No more frames. Closing...")
            break
    
        # Restart the video if we get to the end
        if sourceType.video:
            videoObj.set(vc_pos_frames, 0)
            continue
    
    # .................................................................................................................
    # Resize the frame and add borders
    
    if scaledWH is not None:
        # Resize the image to eventual output size
        scaledFrame = cv2.resize(inFrame, dsize=scaledWH)
    else:
        scaledFrame = inFrame
    
    # Add borders to the frame for drawing 'out-of-bounds'
    scaledFrame = cv2.copyMakeBorder(scaledFrame, 
                                     top=hBorder, 
                                     bottom=hBorder, 
                                     left=wBorder,
                                     right=wBorder,
                                     borderType=solidBorder,
                                     value=borderColor)
    
    
    # .................................................................................................................
    # Draw lines
    
    drawAllLines(scaledFrame, cbData["lineList"], 
                 lineColor=(255, 0, 255), lineThickness=2, circleRadius=5)
    
    drawLineInProgress(scaledFrame, cbData["newPoints"], cbData["mouse"], 
                       lineColor=(255, 255, 0), lineThickness=1, circleRadius=5)
    
    
    # .................................................................................................................
    # Display the frame
    
    cv2.imshow(setupWindow, scaledFrame)        
    
    
    # .................................................................................................................
    # Get key press values
    
    keyPress = cv2.waitKey(frameDelay) & 0xFF
    if (keyPress == ord('q')) | (keyPress == 27) | (keyPress == 32):  # q, Esc or spacebar to close window
        print("")
        print("Key pressed to stop!")
        break
        
        
# ---------------------------------------------------------------------------------------------------------------------
#%% Clean up
    
cv2.destroyAllWindows()
videoObj.release()

# Save video source for possible re-use
saveSourceHistory(historyFile, videoSource)

# ---------------------------------------------------------------------------------------------------------------------
#%% Scrap

