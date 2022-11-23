from WifiComms import WifiRobot
from videoproc import arena
import cv2 as cv
from icecream import ic
import numpy as np
import math

import time

address = "http://192.168.137.18"
headingE = 1
distE = 5

def calibrateMotion(robot: WifiRobot, world: arena, duration):
    pastCoord = world.findRobot()
    robot.tforward(duration)
    time.sleep(duration/1000 + 1)
    currentCoord = world.findRobot()
    
    distanceMoved = np.abs(np.linalg.norm(currentCoord-pastCoord))
    tranMult = distanceMoved/duration

    dy = currentCoord[1]-pastCoord[1]
    dx = currentCoord[0]-pastCoord[0]
    theta0 = math.atan2(dy, dx)
    if theta0<0:
        theta0 += 2*np.pi
    
    pastCoord = world.findRobot()
    robot.tcw(duration)
    time.sleep(duration/1000 + 1)
    robot.tforward(duration)
    time.sleep(duration/1000 + 1)
    currentCoord = world.findRobot()
    
    dy = currentCoord[1]-pastCoord[1]
    dx = currentCoord[0]-pastCoord[0]
    theta1 = math.atan2(dy, dx)
    if theta1<0:
        theta1 += 2*np.pi

    rotated = (theta1 - theta0) / np.pi * 180
    rotMult = rotated/duration
    
    currentHeading = theta1

        
    return (tranMult, rotMult, currentHeading)

def pathFindTo(robot: WifiRobot, world: arena, start, end):
    dy = end[1]-start[1]
    dx = end[0]-start[0]

    vector = math.atan2(dy, dx) * 180/np.pi
    if vector<0:
        vector = vector + 360
    distance = np.abs(np.linalg.norm(end-start))

    deltaHeading = currentHeading - vector
    if np.absolute(deltaHeading) > headingE: #correction needed
        if deltaHeading > 0:
            robot.tcw(np.absolute(deltaHeading) * rotMult)
        else:
            robot.tccw(np.absolute(deltaHeading) * rotMult)
        time.sleep(np.absolute(deltaHeading) * rotMult / 1000 + 1)

    currentHeading = vector

    robot.tforward(distance * tranMult)
    time.sleep(distance * tranMult/1000 + 1)

    if(np.absolute(np.linalg.norm(end-world.findRobot())) > distE):
        pathFindTo(robot, world, world.findRobot(), end) #recursively refine if needed
    
    else:
        return

def goToTunnel(robot: WifiRobot, world: arena):
    tunnelXY = world.t0XY
    robotXY = world.findRobot()
    distance = np.abs(np.linalg.norm(tunnelXY-robotXY))
    if(np.abs(np.linalg.norm(world.t1XY-robotXY))) < distance: # if tunnel 1 is nearer, go to it
        tunnelXY = world.t1XY
    
    pathFindTo(robot, world, robotXY, tunnelXY)
   


if __name__ == "__main__":

    robot = WifiRobot(address)
    world = arena("http://localhost:8081/stream/video.mjpeg")

    tranMult, rotMult, currentHeading = calibrateMotion(robot, world, 1000) #calibrate with 1000ms

    goToTunnel(robot, world)
