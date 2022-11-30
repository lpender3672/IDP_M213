from WifiComms import WifiRobot
from videoproc import arena
import cv2 as cv
from icecream import ic
import numpy as np
import math

import time

address = "http://192.168.137.40"
headingE = 2
distE = 15
maxGo = 20000
blockTresh = 758 #less dense is higher

def deltaAngle(t, c):
    norm = abs(t - c)
    delta = min(norm, 360-norm)

    if (norm > 180):
        if t > c:
            return delta*(-1)
        else:
            return delta
        
    else:
        if t > c:
            return delta
        else:
            return delta*(-1)


def calibrateMotion(robot: WifiRobot, world: arena, duration):
    px,py,po = world.findRobotAruco()
    robot.tforward(duration)
    time.sleep(duration/1000 + 1)
    cx,cy,co = world.findRobotAruco()
    
    distanceMoved = math.dist((px, py), (cx, cy))
    tranMult = distanceMoved/duration

    time.sleep(1)

    robot.tcw(duration)
    time.sleep(duration/1000 + 1)
    dx, dy, do = world.findRobotAruco()
    rotation = do-co
    rotMultCW = rotation / duration

    time.sleep(1)
    robot.tccw(duration)
    time.sleep(duration/1000 + 1)
    ex, ey, eo = world.findRobotAruco()
    rotation = do-eo
    rotMultCCW = rotation / duration
    
        
    return (tranMult, rotMultCW, rotMultCCW)

def correctHeading(robot: WifiRobot, world: arena, targetHeading):
    ic(targetHeading)
    data = world.findRobotAruco()
    while data is None:
        robot.tbackward(1000)
        robot.tforward(2000)
        time.sleep(2)
        data = world.findRobotAruco()
    px, py, po = data

    deltaHeading = deltaAngle(targetHeading, po)
    ic(deltaHeading)
    if deltaHeading > 0:
        robot.tcw(abs(deltaHeading) / rotMultCW)
        time.sleep(abs(deltaHeading) / rotMultCW / 1000 + 0.1)
    else: 
        robot.tccw((abs(deltaHeading)) / rotMultCCW)
        time.sleep(abs(deltaHeading) / rotMultCCW / 1000 + 0.1)

    # # deltaHeading = min(deltaHeading, 360+deltaHeading) #watch this line
    # if deltaHeading > 180:
    #     deltaHeading = 180-deltaHeading
    # ic(deltaHeading)

    # # if abs(deltaHeading) > headingE: #correction needed
    # if deltaHeading > 0:
    #     robot.tcw(abs(deltaHeading) / rotMultCW)
    #     time.sleep(abs(deltaHeading) / rotMultCW / 1000 + 0.1)
    # else:
    #     robot.tccw((abs(deltaHeading)) / rotMultCCW)
    #     time.sleep(abs(deltaHeading) / rotMultCCW / 1000 + 0.1)

    # _, _, po = world.findRobotAruco()
    data = world.findRobotAruco()
    while data is None:
        robot.tbackward(1000)
        robot.tforward(2000)
        time.sleep(2)
        data = world.findRobotAruco()
    px, py, po = data

    deltaHeading = targetHeading - po
    # if deltaHeading > 180:
    #     deltaHeading = 180-deltaHeading
    if abs(deltaHeading) > headingE: #correction needed
        correctHeading(robot, world, targetHeading)





def pathFindTo(robot: WifiRobot, world: arena, end):

    # px, py, po = world.findRobotAruco()
    ic(end)
    data = world.findRobotAruco()
    while data is None:
        robot.tbackward(1000)
        robot.tforward(2000)
        time.sleep(2)
        data = world.findRobotAruco()
    px, py, po = data


    dy = end[1]-py
    dx = end[0]-px
    currentHeading = po

    vector = math.atan2(dy, dx) * 180 / np.pi
    if vector<0:
        vector = vector + 360
    distance = math.dist(end, (px, py))
    ic(distance)

    correctHeading(robot, world, vector)

    dur = min([distance/tranMult, maxGo])
    robot.tforward(dur)
    time.sleep(dur/1000 + 0.2)

    data = world.findRobotAruco()
    while data is None:
        robot.tforward(1000)
        time.sleep(1)
        data = world.findRobotAruco()
    cx, cy, co = data

    ic(cx,cy,co)
    if(math.dist((cx, cy), end) > distE):
        pathFindTo(robot, world, end) #recursively refine if needed
    return
    
    # def goToTunnel(robot: WifiRobot, world: arena):
#     tunnelXY = world.t0XY
#     robotXY = world.findRobot()
#     distance = np.abs(np.linalg.norm(tunnelXY-robotXY))
#     if(np.abs(np.linalg.norm(world.t1XY-robotXY))) < distance: # if tunnel 1 is nearer, go to it
#         tunnelXY = world.t1XY
    
#     pathFindTo(robot, world, robotXY, tunnelXY)
   


if __name__ == "__main__":

    robot = WifiRobot(address)
    world = arena("http://localhost:8081/stream/video.mjpeg")


    robot.drop()
    robot.grn(0)
    robot.red(0)

    robot.amb(1)

    tranMult, rotMultCW, rotMultCCW= calibrateMotion(robot, world, 1000) #calibrate with 1000ms
    ic(tranMult)
    ic(rotMultCW, rotMultCCW)
    time.sleep(2)

    # correctHeading(robot, world, 350)

    robot.drop()
    robot.grn(0)
    robot.red(0)

    pathFindTo(robot, world, world.rampStart)
    pathFindTo(robot, world, world.rampEnd)
    pathFindTo(robot, world, world.b3)
    correctHeading(robot, world, 90)
    robot.drop()
    robot.tforward(2000)
    time.sleep(1)
    robot.pick()
    robot.drop()
    robot.pick()
    robot.drop()
    robot.pick()
    
    rdg = robot.poll()
    if rdg >= blockTresh:
        robot.red(1)
    else:
        robot.grn(1)
    # time.sleep(1)
    # pathFindTo(robot, world, world.arenaCenterR)
    pathFindTo(robot, world, world.arenaCenterG)
    pathFindTo(robot, world, world.tunnelEnd)
    pathFindTo(robot, world, world.tunnelStart)
    if rdg >= blockTresh:
        pathFindTo(robot, world, world.redBox)
    else:
        pathFindTo(robot, world, world.greenBox)
    # pathFindTo(robot, world, world.greenBox)
    correctHeading(robot, world, 270)
    robot.drop()
    robot.tbackward(2000)
    pathFindTo(robot, world, world.start)
    robot.grn(0)
    robot.red(0)
    robot.amb(0)

    # #get second block
    # pathFindTo(robot, world, world.rampStart)
    # pathFindTo(robot, world, world.rampEnd)
    # pathFindTo(robot, world, world.b2)
    # correctHeading(robot, world, 90)
    # robot.drop()
    # robot.tforward(2000)
    # time.sleep(1)
    # robot.pick()
    # robot.drop()
    # robot.pick()
    # robot.drop()
    # robot.pick()
    
    # rdg = robot.poll()
    # if rdg >= blockTresh:
    #     robot.red(1)
    # else:
    #     robot.grn(1)
    # # time.sleep(1)
    # # pathFindTo(robot, world, world.arenaCenterR)
    # pathFindTo(robot, world, world.arenaCenterG)
    # pathFindTo(robot, world, world.tunnelEnd)
    # pathFindTo(robot, world, world.tunnelStart)
    # if rdg >= blockTresh:
    #     pathFindTo(robot, world, world.redBox)
    # else:
    #     pathFindTo(robot, world, world.greenBox)
    # # pathFindTo(robot, world, world.greenBox)
    # correctHeading(robot, world, 270)
    # robot.drop()
    # robot.tbackward(2000)
    # pathFindTo(robot, world, world.start)
    # robot.grn(0)
    # robot.red(0)

    # #get third block
    # pathFindTo(robot, world, world.rampStart)
    # pathFindTo(robot, world, world.rampEnd)
    # pathFindTo(robot, world, world.b1)
    # correctHeading(robot, world, 90)
    # robot.drop()
    # robot.tforward(2000)
    # time.sleep(1)
    # robot.pick()
    # robot.drop()
    # robot.pick()
    # robot.drop()
    # robot.pick()
    
    # rdg = robot.poll()
    # if rdg >= blockTresh:
    #     robot.red(1)
    # else:
    #     robot.grn(1)
    # # time.sleep(1)
    # # pathFindTo(robot, world, world.arenaCenterR)
    # # pathFindTo(robot, world, world.arenaCenterG)
    # pathFindTo(robot, world, world.tunnelEnd)
    # pathFindTo(robot, world, world.tunnelStart)
    # if rdg >= blockTresh:
    #     pathFindTo(robot, world, world.redBox)
    # else:
    #     pathFindTo(robot, world, world.greenBox)
    # # pathFindTo(robot, world, world.greenBox)
    # correctHeading(robot, world, 270)
    # robot.drop()
    # robot.tbackward(2000)
    # pathFindTo(robot, world, world.start)
    # robot.grn(0)
    # robot.red(0)

    # correctHeading(robot, world, 10)
    # correctHeading(robot, world, 350) 
    # correctHeading(robot, world, 10)
    # correctHeading(robot, world, 350) 
    # correctHeading(robot, world, 180)
    # correctHeading(robot, world, 181) 
    # correctHeading(robot, world, 179)
    # correctHeading(robot, world, 180)
    # correctHeading(robot, world, 181) 
    # correctHeading(robot, world, 179)



    # goToTunnel(robot, world)
