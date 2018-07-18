#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 09:17:11 2018

@author: eo
"""

import os
import cv2
import numpy as np

from local.lib.drawlib import loadImageResource, loadFromHistory, saveSourceHistory
from local.lib.windowing import SimpleWindow, breakByKeypress, arrowKeys
    
try:
    import cv.transform.perspective
    import cv.util.functions
    import cv.util.geometry
except ImportError:
    print("")
    print("cv library files not found!")
    print("Library folder must be included in project directory")
    print("")
    print("Requires:")
    print("  cv/transform/perspective")
    print("  cv/util/functions")
    print("  cv/util/geometry")
    print("")
    # Crash to stop spyder without kernel restart or quit nicely if not using spyder...
    if any('SPYDER' in name for name in os.environ): raise SystemExit()
    quit()  # Works nicely everywhere else! 


# Hacky 'global' definition of perspective variables...
xs = 1.0
x1 = 0.0
x2 = 0.0
y1 = 0.5
y2 = 0.5

# ---------------------------------------------------------------------------------------------------------------------
#%% Define functions
    
# Convenience for closing OCV windows interactively
closeall = cv2.destroyAllWindows

# .....................................................................................................................

def drawQuadOutline(displayFrame, quadPoints, lineColor, lineThickness, circleRadius=3, isClosed=True):
    cv2.polylines(displayFrame, [quadPoints], isClosed, lineColor, lineThickness)
    cv2.circle(displayFrame, tuple(quadPoints[0]), 2*circleRadius, lineColor, -1)
    for eachPoint in quadPoints[1:]:
        cv2.circle(displayFrame, tuple(eachPoint), circleRadius, lineColor, -1)

# .....................................................................................................................

def distort_u(u):
    # Compress horizontal space near the edges of the image to correct for
    # a "bending outward" distortion.
    '''
    u = cv.util.functions.symmetric_gain(u, 0.75)
    '''
    u = cv.util.functions.symmetric_gain(u, xs)

    # Extend interpolation range to be between -1 and 2 (from 0 to 1),
    # in order to capture additional horizontal image space in the
    # transformation.
    '''
    return (2 * u) + (-1 * (1 - u))
    '''
    leftX = 0 - x1
    rightX = 1 + x2
    return rightX*u + leftX*(1 - u)

# .....................................................................................................................

def distort_v(v):
    # A rectified perspective image may appear compressed at the top and
    # stretched at the bottom, depending on its specific perspective.
    # This implementation tries to correct this distortion by varying the
    # interpolation factor, via two blended bias coefficients, near and
    # far. This in turn results in the top of the image to be stretched and
    # the bottom compressed.
    '''
    return ((cv.util.functions.fast_bias(v, 0.4) * v) +
            (cv.util.functions.fast_bias(v, 0.3) * (1 - v)))
    '''    
    return ((cv.util.functions.fast_bias(v, y2) * v) +
            (cv.util.functions.fast_bias(v, (1-y1)) * (1 - v)))
    
# .....................................................................................................................

def updatePerspective(dictHolder, lowRes=False):
    
    # Set lower resolution if needed, which helps speed up interactivity
    scaling = 0.5 if lowRes else 1
    
    # Get quad points without border offsets
    newQuadPoints = dictHolder["quadPoints"] - dictHolder["borderWH"]
            
    # Update perspective transform
    dictHolder["persp"] = cv.transform.perspective.RectifyTransform(*newQuadPoints, 
                                                                    distort_u_fn=distort_u, 
                                                                    distort_v_fn=distort_v,
                                                                    scale_w=scaling, 
                                                                    scale_h=scaling)
# .....................................................................................................................

def snapToBorder(paramData, pointSelect):

    # For convenience
    frameW, frameH = paramData["frameWH"] - np.array((1,1))
    px, py = paramData["quadPoints"][pointSelect] - paramData["borderWH"]
    
    # Set the threshold for how close a point can be 'inside' the frame and still be snapped
    innerThreshold = 0.05*min(frameW, frameH)
    
    # Check if point is snappable in x
    snapXleft = (px < 0) or (0 < px <= innerThreshold)
    snapXright = (px > frameW) or (0 < (frameW - px) <= innerThreshold)
    if snapXleft: px = 0    
    if snapXright: px = frameW
    
    # Check if point is snappable in x
    snapYleft = (py < 0) or (0 < py <= innerThreshold)
    snapYright = (py > frameH) or (0 < (frameH - py) <= innerThreshold)
    if snapYleft: py = 0    
    if snapYright: py = frameH
    
    # Update the quad points if any of the co-ords were snapped
    if any([snapXleft, snapXright, snapYleft, snapYright]):
        paramData["quadPoints"][pointSelect] = np.array((px,py)) + paramData["borderWH"]

# .....................................................................................................................
    
def getSimpleQuad(frameWH, borderWH):
    
    # Create quad that covers the whole frame evenly
    initialQuad = np.array([(0, 0), (frameWH[0] - 1, 0), 
                            (frameWH[0]-1, frameWH[1]-1), (0, frameWH[1]-1)], dtype=np.int32)
    initialQuad = initialQuad + np.array(borderWH)
    return initialQuad

# .....................................................................................................................

def quadPrintOut(paramData):
    
    # Calculate normalized values
    quadOffset = paramData["quadPoints"] - paramData["borderWH"]
    quadNormalized = quadOffset/(paramData["frameWH"] - np.array((1,1)))
    
    # Build output strings
    quadOffsetInnerString = ["[{:.0f}, {:.0f}]".format(*eachPoint) for eachPoint in quadOffset]
    normInnerString = ["[{:.4f}, {:.4f}]".format(*eachPoint) for eachPoint in quadNormalized]    
    quadOffsetString = "[" + ", ".join(quadOffsetInnerString) + "]"
    normString = "[" + ", ".join(normInnerString) + "]"
    
    # Print out line data that works for config file
    print("")
    print("")
    print("****************************************")
    print("********* PERSPECTIVE SETTINGS *********")
    print("****************************************")
    print("")
    print("Using frame dimensions:", paramData["frameWH"][0], "x", paramData["frameWH"][1])
    print("")
    print("\tPixel values:")
    print("quad: ", quadOffsetString)
    print("")
    print("\tNormalized values:")
    print("quad: ", normString)

# .....................................................................................................................

def overlayGrid(displayFrame, xGap=20, yGap=20, lineColor=(220,220,220), lineThickness=1):
    
    # Get size of input frame to figure out how many lines to draw
    frameW, frameH = displayFrame.shape[1::-1] - np.array((1,1))
    
    # Get offsets in case the grid doesn't fit nicely into the frame
    xoffset = int(np.remainder(frameW, xGap)/2)
    yoffset = int(np.remainder(frameH, yGap)/2)
    
    # Draw horizontal lines
    numY = int(np.ceil(frameH/yGap))
    yPixelIdx = [yoffset + yIdx*yGap for yIdx in range(numY)]
    for eachYPx in yPixelIdx:
        cv2.line(displayFrame, (-15, eachYPx), (frameW + 15, eachYPx), lineColor, lineThickness)
        
    # Draw vertical lines
    numX = int(np.ceil(frameW/xGap))
    xPixelIdx = [xoffset + xIdx*xGap for xIdx in range(numX)]
    for eachXPx in xPixelIdx:
        cv2.line(displayFrame, (eachXPx, -15), (eachXPx, frameH + 15), lineColor, lineThickness)

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
        for pointIdx, eachPoint in enumerate(param["quadPoints"]):
            
            # Calculate the distance between mouse and point
            distSq = np.sum(np.square(mxy - eachPoint))
            
            # Record the closest point
            if distSq < minSqDist:
                minSqDist = distSq
                bestMatchIdx = pointIdx
    
        # Figure out if we need to change the point position
        distanceThreshold = 50**2            
        param["pointHover"] = bestMatchIdx if minSqDist < distanceThreshold else None 
    
    # .................................................................................................................
    # Select nearest point on left click
    
    if event == cv2.EVENT_LBUTTONDOWN:

        # Select the point nearest to the mouse before clicking
        param["pointSelect"] = param["pointHover"]  
        
    # .................................................................................................................
    # Move points on middle click & drag
    
    #print(flags)
    if flags == (mouseMoveOffset + cv2.EVENT_FLAG_LBUTTON):
        
        # Update the dragged points based on the mouse position
        pointSelect = param["pointSelect"]
        if pointSelect is not None:
            
            # Replace the old point co-ordinates with the new ones (after dragging)
            param["quadPoints"][pointSelect] = mxy
            updatePerspective(param, lowRes=True)
        
    # .................................................................................................................
    # Update with high-res image after finished dragging
    
    if event == cv2.EVENT_LBUTTONUP:
        updatePerspective(param, lowRes=False)
        
    # .................................................................................................................
    # Reset quad with right-click
    
    if event == cv2.EVENT_RBUTTONDOWN:
        param["quadPoints"] = getSimpleQuad(param["frameWH"], param["borderWH"])
        updatePerspective(param, lowRes=False)

    # .................................................................................................................
    
    
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

# Create object storing perspective transformation info
initialQuad = getSimpleQuad(scaledWH, (wBorder, hBorder))
perspective = cv.transform.perspective.RectifyTransform(*initialQuad, 
                                                        distort_u_fn=distort_u, 
                                                        distort_v_fn=distort_v,
                                                        scale_w=1, 
                                                        scale_h=1)

# Initialize a variable used to pass data to mouse callback
cbData = {"mouse": (0, 0),  
          "quadPoints": initialQuad,
          "pointSelect": None,
          "pointHover": None,
          "frameWH": np.array(scaledWH), 
          "borderWH": np.array((wBorder, hBorder)),
          "persp": perspective}


# ---------------------------------------------------------------------------------------------------------------------
#%% Print out instructions

print("")
print("----------------------------------------------------------------")
print("------------------------- Instructions -------------------------")
print("----------------------------------------------------------------")
print("")
print("Left click:")
print("  - Drag quadrilateral points around")
print("")
print("Right click:")
print("  - Reset perspective quadrilateral")
print("")
print("Keypress p:")
print("  - Print out current quad. settings")
print("")
print("Keypress b:")
print("  - Snap points near mouse to frame border")
print("")
print("Keypress g:")
print("  - Toggle grid line overlay")
print("")
print("Spacebar:")
print("  - Rotate/re-assign top left corner point")
print("")
print("Arrow keys:")
print("  - Nudge points near to mouse cursor")
print("")
print("Exiting:")
print("  - Use q or Esc to close the window")
print("")
print("----------------------------------------------------------------")
print("----------------------------------------------------------------")

# ---------------------------------------------------------------------------------------------------------------------
#%% Run video

# For convenience
leftArrow, upArrow, rightArrow, downArrow = 81, 82, 83, 84
arrowKeyList = [leftArrow, upArrow, rightArrow, downArrow]

# Set up grid sizing
gridSizes = [None, 100, 50, 20, 10, 5]
gridIndex = 0

# Set up main window
drawWindow = SimpleWindow("Draw Persp. Quadrilateral")
drawWindow.addCallback(mouseCallback, cbData)

# Set up masking window
warpWindow = SimpleWindow("Transformed Image")

# Set up trackbar control window
controlWindow = SimpleWindow("Controls")

# Set up trackbars
xsRef = "xs"
x1Ref, x2Ref = "x1", "x2"
y1Ref, y2Ref = "y1", "y2"
controlWindow.addTrackbar(xsRef, 100, 200)
controlWindow.addTrackbar(x1Ref, 0, 100)
controlWindow.addTrackbar(x2Ref, 0, 100)
controlWindow.addTrackbar(y1Ref, 50, 100)
controlWindow.addTrackbar(y2Ref, 50, 100)
prevpUpdate, currpUpdate = True, False
controlWindow.imshow(np.zeros((1, scaledWH[0], 3), dtype=np.uint8))     # Use a blank image for the control window

# Position windows
leftSpacing = 100
topSpacing = 150
drawWindow.move(x = leftSpacing, y = topSpacing)
warpWindow.move(x = leftSpacing + wBorder + scaledWH[0] + leftSpacing, y = topSpacing)
controlWindow.move(x = leftSpacing + wBorder + scaledWH[0] + leftSpacing, y = topSpacing + targetWH[1] + 1*hBorder)

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
            break
            videoObj.set(vc_pos_frames, 0)
            continue
    
    # .................................................................................................................
    # Resize the frame and add borders
    
    if scaledWH is not None:
        # Resize the image to eventual output size
        scaledFrame = cv2.resize(inFrame, dsize=scaledWH)
    else:
        scaledFrame = inFrame
        
    # Add grid lines
    gridGap = gridSizes[gridIndex]
    if gridGap is not None:
        overlayGrid(scaledFrame, gridGap, gridGap)
    
    # Add borders to the frame for drawing 'out-of-bounds'
    borderedFrame = cv2.copyMakeBorder(scaledFrame, 
                                       top=hBorder, 
                                       bottom=hBorder, 
                                       left=wBorder,
                                       right=wBorder,
                                       borderType=solidBorder,
                                       value=borderColor)    
    
    # .................................................................................................................
    # Draw perspective region and read sliders
    
    drawQuadOutline(borderedFrame, cbData["quadPoints"], lineColor=(0,255,255), lineThickness=1)
    currpUpdate = False
    
    # Read track bar values
    xsChanged, xsRead = controlWindow.readTrackbar(xsRef)
    x1Changed, x1Read = controlWindow.readTrackbar(x1Ref)
    x2Changed, x2Read = controlWindow.readTrackbar(x2Ref)
    y1Changed, y1Read = controlWindow.readTrackbar(y1Ref)
    y2Changed, y2Read = controlWindow.readTrackbar(y2Ref)
    
    # Update trackbars only when they change
    if xsChanged:
        xs = xsRead/100
        updatePerspective(cbData, lowRes=True)
        currpUpdate = True
    
    if x1Changed or x2Changed: #x1Read != oldx1 or x2Read != oldx2:
        x1 = x1Read/100
        x2 = x2Read/100
        updatePerspective(cbData, lowRes=True)
        currpUpdate = True
    
    if y1Changed or y2Changed:# y1Read != oldy1 or y2Read != oldy2:
        y1 = min(max(0.01, y1Read/100), 0.99)
        y2 = min(max(0.01, y2Read/100), 0.99)
        updatePerspective(cbData, lowRes=True)
        currpUpdate = True
    
    # .................................................................................................................
    # Get transformed data and scale it for display
    
    # Get perspective transform based on gui quad co-ordinates
    transformedFrame = cbData["persp"].transform_image(scaledFrame)
    transformedWH = transformedFrame.shape[1::-1]
    
    # Scale up/down the frame to match a target size
    frameScaling = 1 / max(np.array(transformedWH) / np.array(targetWH))
    warpScaled = cv2.resize(transformedFrame, dsize=None, fx=frameScaling, fy=frameScaling)
    warpWH = warpScaled.shape[1::-1]
    
    # Figure out width padding
    widthGap, heightGap = targetWH[0] - warpWH[0], targetWH[1] - warpWH[1]
    leftPad, topPad = int(widthGap/2), int(heightGap/2)
    rightPad, botPad = widthGap - leftPad, heightGap - topPad
    
    # Add width/height padding to keep the frame at a consistent size
    warpPadFrame = cv2.copyMakeBorder(warpScaled, 
                                      top=topPad, 
                                      bottom=botPad, 
                                      left=leftPad,
                                      right=rightPad,
                                      borderType=solidBorder,
                                      value=(0,0,0))    
    
    
    # .................................................................................................................
    # Display
    
    # Show main drawing window
    winExists = drawWindow.imshow(borderedFrame)
    if not winExists: break
    
    # Show perspective transformed image
    winExists = warpWindow.imshow(warpPadFrame)
    if not winExists: break 
    
    # .................................................................................................................
    # Get key press values
    
    # Get keypress & close window if q/Esc are pressed
    reqBreak, keyPress = breakByKeypress(frameDelay)
    if reqBreak: break
    
    # Print out quad info when p is press
    if (keyPress == ord('p')) or (keyPress == ord('P')):
        quadPrintOut(cbData)
    
    # Snap nearest point to the frame border (if it is close-ish)
    if (keyPress == ord('b') or (keyPress == ord('B'))):
        pointHover = cbData["pointHover"]
        if pointHover is not None:
            snapToBorder(cbData, pointHover)
            updatePerspective(cbData, lowRes=False)
    
    # Update grid overlay
    if (keyPress == ord('g') or (keyPress == ord('G'))):
        gridIndex = (gridIndex + 1) % len(gridSizes)
    
    # Rotate corner order when spacebar is pressed
    if keyPress == 32:
        cbData["quadPoints"] = np.roll(cbData["quadPoints"], -1, axis=0)
        updatePerspective(cbData, lowRes=True)
        currpUpdate = True
        
    # Allow for small adjustments using the arrow keys (adjust last modified point)
    arrowPressed, arrowXY = arrowKeys(keyPress)
    if arrowPressed:
        pointHover = cbData["pointHover"]
        if pointHover is not None:
            cbData["quadPoints"][pointHover] += arrowXY
            updatePerspective(cbData, lowRes=True)
            currpUpdate = True
            
    # Perform high quality perspective update when we're done changing the trackbars/arrow keys
    if prevpUpdate and not currpUpdate:
        updatePerspective(cbData, lowRes=False)
    prevpUpdate = currpUpdate
            
        
        
# ---------------------------------------------------------------------------------------------------------------------
#%% Clean up
    
cv2.destroyAllWindows()
videoObj.release()

# Final print out, just in case
quadPrintOut(cbData)

# Save video source for possible re-use
saveSourceHistory(historyFile, videoSource)


# ---------------------------------------------------------------------------------------------------------------------
#%% Scrap

