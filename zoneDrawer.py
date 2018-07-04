#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 10:17:50 2018

@author: eo
"""

import os
import cv2
import numpy as np

from local.lib.drawlib import loadImageResource, loadFromHistory, saveSourceHistory

# ---------------------------------------------------------------------------------------------------------------------
#%% Define functions
    
# Convenience for closing OCV windows interactively
closeall = cv2.destroyAllWindows

# .....................................................................................................................

def drawZone(displayFrame, zonePoints, lineColor, lineThickness, circleRadius=3, isClosed=True):
    cv2.polylines(displayFrame, [zonePoints], isClosed, lineColor, lineThickness)
    for eachPoint in zonePoints:
        cv2.circle(displayFrame, tuple(eachPoint), circleRadius, lineColor, -1)

# .....................................................................................................................

def drawAllZones(displayFrame, inZones, lineColor, lineThickness):
    for eachZone in inZones:
        drawZone(displayFrame, eachZone, lineColor, lineThickness)
    
# .....................................................................................................................

def drawZoneInProgress(displayFrame, pointsInProgress, mousePoint, lineColor, lineThickness, circleRadius):
    
    numPoints = len(pointsInProgress)
    
    # If there are no points, don't try to do anything
    if numPoints == 0:
        return
    
    # If only one point exists, just draw a line since we can't draw a polygon
    if numPoints == 1:
        cv2.line(displayFrame, tuple(pointsInProgress[0]), tuple(mousePoint), lineColor, lineThickness)
    
    # If there are multiple zone points, connect them together (without closing the polygon)
    if numPoints > 1:
        drawArray = np.vstack((pointsInProgress, mousePoint))
        drawZone(displayFrame, drawArray, lineColor, lineThickness, isClosed=True)
    
    # Draw circles at each of the new points to highlight them
    for eachPoint in pointsInProgress:
        cv2.circle(displayFrame, tuple(eachPoint), circleRadius, lineColor, -1)

# .....................................................................................................................
        
def zonePrintOut(newZone, frameWH, borderWH, updating=False):
    
    # Calculate values after removing border offset
    zoneOffset = newZone - borderWH
    
    # Calculate normalized values
    normalizedZone = zoneOffset/(frameWH - np.array((1,1)))
    
    # Build output strings
    zoneOffsetInnerString = ["[{:.0f}, {:.0f}]".format(*eachPoint) for eachPoint in zoneOffset]
    normZoneInnerString = ["[{:.4f}, {:.4f}]".format(*eachPoint) for eachPoint in normalizedZone]    
    zoneOffsetString = "[" + ", ".join(zoneOffsetInnerString) + "]"
    normZoneString = "[" + ", ".join(normZoneInnerString) + "]"
    
    # Print out zone data that works for config file
    print("")
    print("")
    print("****************************************")
    
    if updating:
        print("************* UPDATED ZONE *************")
    else:
        print("*************** NEW ZONE ***************")
        
    print("****************************************")
    print("")
    print("Using frame dimensions:", frameWH[0], "x", frameWH[1])
    print("")
    print("\tPixel values:")
    print("corners: ", zoneOffsetString)
    print("")
    print("\tNormalized values:")
    print("corners: ", normZoneString)
    
# .....................................................................................................................
    
# ---------------------------------------------------------------------------------------------------------------------
#%% Define callback
    
def mouseCallback(event, mx, my, flags, param):
    
    # Record mouse x and y positions at all times (in case we need to draw zone-in-progress)
    mxy = np.array((mx, my))
    param["mouse"] = mxy
    
    
    # .................................................................................................................
    # Add points with left click
    
    if event == cv2.EVENT_LBUTTONDOWN:
        
        # Add the clicked point to the list of new points
        param["newPoints"].append(mxy)        
        
        
    # .................................................................................................................
    # Clear zones with right-click
    
    if event == cv2.EVENT_RBUTTONDOWN:
        
        # Clear zones that are moused over, but only if we aren't currently drawing a new zone
        if len(param["newPoints"]) == 0:
            param["zoneList"] = [eachZone for eachZone in param["zoneList"] if (cv2.pointPolygonTest(eachZone, tuple(mxy), measureDist=False) < 0)]
            
        # Clear zone-in-progress points regardless
        param["newPoints"] = []
    
    
    # .................................................................................................................
    # Select nearest zone-points on middle-down
    
    if event == cv2.EVENT_MBUTTONDOWN and len(param["newPoints"]) == 0:
        
        # Check for the closest point
        minSqDist = 1E9
        bestMatchIdx = -1
        for zoneIdx, eachZone in enumerate(param["zoneList"]):            
            for pointIdx, eachPoint in enumerate(eachZone):
                
                # Calculate the distance between mouse and point
                distSq = np.sum(np.square(mxy - eachPoint))
                
                # Record the closest point
                if distSq < minSqDist:
                    minSqDist = distSq
                    bestMatchIdx = (zoneIdx, pointIdx)

        # Figure out if we need to change the point position
        distanceThreshold = 50**2            
        param["zonePointSelect"] = bestMatchIdx if minSqDist < distanceThreshold else None
        
    
    # .................................................................................................................
    # Close zone on middle-down
    
    if event == cv2.EVENT_MBUTTONDOWN and len(param["newPoints"]) > 0:
        
        # Only create a zone if there are 3 or more points
        if len(param["newPoints"]) > 2:
            
            # Convert to int32 numpy array for drawing purposes
            newZonePoints = np.array(param["newPoints"], dtype=np.int32)
            
            # Add a new zone to the list
            param["zoneList"].append(newZonePoints)
            
            # Clear the points used to create the zone so we don't re-use them
            param["newPoints"] = []    
            
            # Finally, print out the info so it can be used for something!
            zonePrintOut(newZonePoints, param["frameWH"], param["borderWH"], updating=False)
            
    # .................................................................................................................
    # Move points on middle click & drag
    
    #print(event, flags)
    if flags == (mouseMoveOffset + cv2.EVENT_FLAG_MBUTTON):
        # Don't do anything if a zone is in-progress
        if len(param["newPoints"]) > 0:
            return
        
        # Update the dragged points based on the mouse position
        if param["zonePointSelect"] is not None:
            
            # Get selection indices for convenience
            zoneSelect, pointSelect = param["zonePointSelect"]
            
            # Replace the old point co-ordinates with the new ones (after dragging)
            param["zoneList"][zoneSelect][pointSelect] = mxy
        
        
    # .................................................................................................................
    # Release dragged zone points on middle-up
    
    if event == cv2.EVENT_MBUTTONUP:
        
        # Print out the new zone info if it had a point dragged
        if param["zonePointSelect"] is not None:            
            zonePrintOut(param["zoneList"][zoneSelect], param["frameWH"], param["borderWH"], updating=True)
        
        # Reset the point selection
        param["zonePointSelect"] = None
    
    
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
          "zoneList": [], 
          "zonePointSelect": None,
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
print("  - No limit to number of points defining a zone")
print("  - Large number of points may incur a performance hit however")
print("  - Use middle click to finish zone")
print("")
print("Right click:")
print("  - Clear a zone-in-progress (if currently drawing)")
print("  - Otherwise, clear zone based on mouse position")
print("")
print("Middle click:")
print("  - Complete zone drawing (if currently drawing)")
print("  - Otherwise, drag/move points")
print("  - Updated zone coordinates will be printed out on release")
print("")
print("Exiting:")
print("  - Use q, Esc or spacebar to close the window")
print("")
print("----------------------------------------------------------------")
print("----------------------------------------------------------------")


# ---------------------------------------------------------------------------------------------------------------------
#%% Run video

# Set up window
setupWindow = "Draw Zones"
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
    # Draw zones
    
    drawAllZones(scaledFrame, cbData["zoneList"],
                 lineColor=(255,0,255), lineThickness=2)
    
    drawZoneInProgress(scaledFrame, cbData["newPoints"], cbData["mouse"], 
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

