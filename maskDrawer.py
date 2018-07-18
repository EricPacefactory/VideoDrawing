#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 12:42:30 2018

@author: eo
"""

import os
import cv2
import numpy as np

from local.lib.drawlib import loadImageResource, loadFromHistory, saveSourceHistory, guiConfirm, guiSave
from local.lib.windowing import SimpleWindow, breakByKeypress, arrowKeys

# ---------------------------------------------------------------------------------------------------------------------
#%% Define functions
    
# Convenience for closing OCV windows interactively
closeall = cv2.destroyAllWindows

# .....................................................................................................................

def drawMaskOutline(displayFrame, maskPoints, lineColor, lineThickness, circleRadius=3, isClosed=True):
    cv2.polylines(displayFrame, [maskPoints], isClosed, lineColor, lineThickness)
    for eachPoint in maskPoints:
        cv2.circle(displayFrame, tuple(eachPoint), circleRadius, lineColor, -1)

# .....................................................................................................................

def drawAllMaskOutlines(displayFrame, inMasks, lineColor, lineThickness):
    for eachMask in inMasks:
        drawMaskOutline(displayFrame, eachMask, lineColor, lineThickness)
    
# .....................................................................................................................

def drawMaskInProgress(displayFrame, pointsInProgress, mousePoint, lineColor, lineThickness, circleRadius):
    
    numPoints = len(pointsInProgress)
    
    # If there are no points, don't try to do anything
    if numPoints == 0:
        return
    
    # If only one point exists, just draw a line since we can't draw a polygon
    if numPoints == 1:
        cv2.line(displayFrame, tuple(pointsInProgress[0]), tuple(mousePoint), lineColor, lineThickness)
    
    # If there are multiple points, connect them together (without closing the polygon)
    if numPoints > 1:
        drawArray = np.vstack((pointsInProgress, mousePoint))
        drawMaskOutline(displayFrame, drawArray, lineColor, lineThickness, isClosed=True)
    
    # Draw circles at each of the new points to highlight them
    for eachPoint in pointsInProgress:
        cv2.circle(displayFrame, tuple(eachPoint), circleRadius, lineColor, -1)

# .....................................................................................................................

def drawMaskImage(maskList, borderWH, invert):    

    # Create base mask image and select appropriate masking color
    if invert:
        maskFrame = np.zeros((scaledWH[1], scaledWH[0], 3), dtype=np.uint8)
        maskColor = (255, 255, 255)
    else:
        maskFrame = np.full((scaledWH[1], scaledWH[0], 3), (255,255,255), dtype=np.uint8)
        maskColor = (0, 0, 0)
    
    # Draw polygons for masking, with border offset taken into account
    for eachMask in maskList:
        
        # Draw mask regions with drawing borders removed
        borderlessMaskRegion = eachMask - borderWH
        cv2.fillPoly(maskFrame, [borderlessMaskRegion], maskColor)
        
        # Draw additional lines around polygons, since there appears to be issues with aliasing
        isClosed = True
        cv2.polylines(maskFrame, [borderlessMaskRegion], isClosed, maskColor, 2)

    return maskFrame

# .....................................................................................................................
    
# ---------------------------------------------------------------------------------------------------------------------
#%% Define callback
    
def mouseCallback(event, mx, my, flags, param):
    
    # Record mouse x and y positions at all times (in case we need to draw mask-in-progress)
    mxy = np.array((mx, my))
    param["mouse"] = mxy
    
    # .................................................................................................................
    # Get point hovering
        
    if flags != (mouseMoveOffset + cv2.EVENT_FLAG_LBUTTON):
        
        # Check for the closest point
        minSqDist = 1E9
        bestMatchIdx = -1
        for maskIdx, eachMask in enumerate(param["maskList"]):            
            for pointIdx, eachPoint in enumerate(eachMask):
                
                # Calculate the distance between mouse and point
                distSq = np.sum(np.square(mxy - eachPoint))
                
                # Record the closest point
                if distSq < minSqDist:
                    minSqDist = distSq
                    bestMatchIdx = (maskIdx, pointIdx)

        # Figure out if we need to change the point position
        distanceThreshold = 50**2            
        param["maskPointHover"] = bestMatchIdx if minSqDist < distanceThreshold else None
    
    # ..................................................................................................................
    # Add points with left click
    
    if event == cv2.EVENT_LBUTTONDOWN:
        
        # Add the clicked point to the list of new points
        param["newPoints"].append(mxy)        
        
        
    # ..................................................................................................................
    # Clear masks with right-click
    
    if event == cv2.EVENT_RBUTTONDOWN:
        
        # Clear masks that are moused over, but only if we aren't currently drawing a new region
        if len(param["newPoints"]) == 0:
            param["maskList"] = [eachMask for eachMask in param["maskList"] if (cv2.pointPolygonTest(eachMask, tuple(mxy), measureDist=False) < 0)]
            
        # Clear mask-in-progress points regardless
        param["newPoints"] = []
    
    
    # ..................................................................................................................
    # Select nearest mask-point on middle-down
    
    if event == cv2.EVENT_MBUTTONDOWN and len(param["newPoints"]) == 0:
        
        param["maskPointSelect"] = param["maskPointHover"]
        
    
    # ..................................................................................................................
    # Close mask on middle-down
    
    if event == cv2.EVENT_MBUTTONDOWN and len(param["newPoints"]) > 0:
        
        # Only create a mask if there are 3 or more points
        if len(param["newPoints"]) > 2:
            
            # Convert to int32 numpy array for drawing purposes
            newMaskPoints = np.array(param["newPoints"], dtype=np.int32)
            
            # Add a new mask to the list
            param["maskList"].append(newMaskPoints)
            
            # Clear the points used to create the mask so we don't re-use them
            param["newPoints"] = []    
            
    # ..................................................................................................................
    # Move points on middle click & drag
    
    #print(flags)
    if flags == (mouseMoveOffset + cv2.EVENT_FLAG_MBUTTON):
        # Don't do anything if a mask is in-progress
        if len(param["newPoints"]) > 0:
            return
        
        # Update the dragged points based on the mouse position
        if param["maskPointSelect"] is not None:
            
            # Get selection indices for convenience
            maskSelect, pointSelect = param["maskPointSelect"]
            
            # Replace the old point co-ordinates with the new ones (after dragging)
            param["maskList"][maskSelect][pointSelect] = mxy
        
        
    # ..................................................................................................................
    # Release dragged mask points on middle-up
    
    if event == cv2.EVENT_MBUTTONUP:
        
        # Reset the point selection
        param["maskPointSelect"] = None
    
    
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

# Masking control variables
maskInvert = False
maskWithImage = True

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
          "maskList": [], 
          "maskPointSelect": None,
          "maskPointHover": None,
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
print("  - Add new points to define mask region")
print("  - Use middle click to finish region")
print("")
print("Right click:")
print("  - Clear a mask-in-progress (if currently drawing)")
print("  - Otherwise, clear regions based on mouse position")
print("")
print("Middle click:")
print("  - Complete mask drawing (if currently drawing)")
print("  - Otherwise, drag/move points")
print("")
print("Arrow keys:")
print("  - Nudge points near to mouse cursor")
print("")
print("Keypress i:")
print("  - Invert mask drawing")
print("")
print("Keypress h:")
print("  - Toggle masking with image")
print("")
print("Exiting:")
print("  - Use q, Esc or spacebar to close the window and save")
print("")
print("----------------------------------------------------------------")
print("----------------------------------------------------------------")


# ---------------------------------------------------------------------------------------------------------------------
#%% Run video

# For convenience
leftArrow, upArrow, rightArrow, downArrow = 81, 82, 83, 84
arrowKeyList = [leftArrow, upArrow, rightArrow, downArrow]

# Position windows
leftSpacing = 100
topSpacing = 150

# Set up main window
drawWindow = SimpleWindow("Draw Mask", 
                          x = leftSpacing, 
                          y = topSpacing)
drawWindow.addCallback(mouseCallback, cbData)

# Set up masking window
maskWindow = SimpleWindow("Masked Image", 
                          x = leftSpacing + wBorder + scaledWH[0] + leftSpacing,
                          y = topSpacing + hBorder/2)

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

    # Resize the image to eventual output size
    scaledFrame = cv2.resize(inFrame, dsize=scaledWH)
    
    # Add borders to the frame for drawing 'out-of-bounds'
    borderedFrame = cv2.copyMakeBorder(scaledFrame, 
                                       top=hBorder, 
                                       bottom=hBorder, 
                                       left=wBorder,
                                       right=wBorder,
                                       borderType=solidBorder,
                                       value=borderColor)    
    
    
    # .................................................................................................................
    # Draw mask regions
    
    drawAllMaskOutlines(borderedFrame, cbData["maskList"],
                        lineColor=(0,255,255), lineThickness=1)
    
    drawMaskInProgress(borderedFrame, cbData["newPoints"], cbData["mouse"], 
                       lineColor=(0, 170, 220), lineThickness=1, circleRadius=5)

    maskFrame = drawMaskImage(cbData["maskList"], cbData["borderWH"], maskInvert)
    
    
    # .................................................................................................................
    # Display
    
    # Show main drawing window
    winExists = drawWindow.imshow(borderedFrame)
    if not winExists: break
    
    # Include the original image if desired
    maskDisplayFrame = cv2.bitwise_and(scaledFrame, maskFrame) if maskWithImage else maskFrame.copy()
        
    # Show masked image
    winExists = maskWindow.imshow(maskDisplayFrame)
    if not winExists: break
    
    
    # .................................................................................................................
    # Get key press values
    
    # Get keypress & close window if q/Esc are pressed
    reqBreak, keyPress = breakByKeypress(frameDelay)
    if reqBreak: break
    
    # Nudge mask points with arrow keys 
    arrowPressed, arrowXY = arrowKeys(keyPress)
    if arrowPressed:
        if cbData["maskPointHover"] is not None:
            maskHover, pointHover = cbData["maskPointHover"]
            cbData["maskList"][maskHover][pointHover] += arrowXY
    
    # Invert mask colors when i is pressed
    if (keyPress == ord('i')) or (keyPress == ord('I')):
        maskInvert = not maskInvert
        
    # Toggle masking with the video image when h is pressed
    if (keyPress == ord('h')) or (keyPress == ord('H')):
        maskWithImage = not maskWithImage
        
        
# ---------------------------------------------------------------------------------------------------------------------
#%% Clean up
    
cv2.destroyAllWindows()
videoObj.release()

# Save video source for possible re-use
saveSourceHistory(historyFile, videoSource)

# ---------------------------------------------------------------------------------------------------------------------
#%% Save mask

# Get user confirmation before trying to save
saveMask = guiConfirm(confirmText="Save mask image?", windowTitle="Save mask")
if saveMask:
    saveMaskSource = guiSave(windowTitle="Save mask", fileTypes=[["image", "*.png"]])
    if saveMaskSource is not None:
        cv2.imwrite(saveMaskSource, maskFrame)
        print("")
        print("Saved mask image:")
        print(saveMaskSource)

# ---------------------------------------------------------------------------------------------------------------------
#%% Scrap

