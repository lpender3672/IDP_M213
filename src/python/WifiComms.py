import requests

from icecream import ic
import time

class WifiRobot:

    tranMult = 1
    rotMult = 1

    def __init__(self, address) -> None:
        self.address = address

    #move forward (time)
    def tforward(self, millis:int):
        params = {
            "go": str(int(millis))
        }
        ic(params)
        requests.get(self.address, params=params)
        time.sleep(0.5)
        return
    
    def tbackward(self, millis:int):
        params = {
            "rev": str(int(millis))
        }
        ic(params)
        requests.get(self.address, params=params)
        time.sleep(0.5)
        return
        
    #rotate (time)
    def tcw(self, millis):
        params = {
            "rcw": str(int(millis))
        }
        ic(params)
        requests.get(self.address, params=params)
        time.sleep(0.5)
        return
        
    def tccw(self, millis):
        params = {
            "rccw": str(int(millis))
        }
        ic(params)
        requests.get(self.address, params=params)
        time.sleep(0.5)
        return

    def pick(self):
        params = {
            "pick": " "
        }
        ic(params)
        requests.get(self.address, params=params)
        time.sleep(0.5)
        return
    
    def drop(self):
        params = {
            "drop": " "
        }
        ic(params)
        requests.get(self.address, params=params)
        time.sleep(0.5)
        return

    def red(self, state):
        params = {
            "r": state
        }
        ic(params)
        requests.get(self.address, params=params)
        time.sleep(0.5)
        return

    def grn(self, state):
        params = {
            "g": state
        }
        ic(params)
        requests.get(self.address, params=params)
        time.sleep(0.5)
        return

    def amb(self, state):
        params = {
            "a": state
        }
        ic(params)
        requests.get(self.address, params=params)
        time.sleep(0.5)
        return

    def poll(self): 
        params = {
            "null": "0"
        }
        ic(params)
        r= requests.get(self.address, params=params)
        ic(int(r.text))
        time.sleep(0.5)
        return (int(r.text))

    

#this section only for testing
if __name__ == "__main__":
    robot = WifiRobot("http://192.168.137.99")

    robot.red(0)
    robot.grn(0)

    # robot.pick()
    # robot.drop()
    # robot.pick()
    # robot.drop()
    # robot.pick()
    # time.sleep(2)
    # rdg = robot.poll()
    # robot.drop()
    # robot.amb(0)

    # blockTresh = 720 #less dense is higher

    # if rdg >= blockTresh:
    #     robot.red(1)
    # else:
    #     robot.grn(1)

    # robot.amb(0)
    # robot.grn(0)
    # robot.red(0)

    # robot.tforward(2000)
    # robot.tcw(1000)
    # robot.tccw(1000)



