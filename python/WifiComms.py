import requests

from icecream import ic
import time

class WifiRobot:

    tranMult = 1
    rotMult = 1

    def __init__(self, address) -> None:
        self.address = address

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

    

 
if __name__ == "__main__":
    robot = WifiRobot("http://192.168.137.236")

    robot.pick()
    robot.drop()
    robot.pick()
    robot.drop()
    robot.pick()
    time.sleep(2)
    robot.poll()
    robot.drop()
    # robot.amb(0)



