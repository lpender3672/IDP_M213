import requests

from icecream import ic

class WifiRobot:

    tranMult = 1
    rotMult = 1

    def __init__(self, address) -> None:
        self.address = address

    def tforward(self, millis:int):
        params = {
            "go": str(int(millis))
        }
        requests.get(self.address, params=params)
        return

    def tcw(self, millis):
        params = {
            "rcw": str(int(millis))
        }
        requests.get(self.address, params=params)
        return
        
    def tccw(self, millis):
        params = {
            "rccw": str(int(millis))
        }
        requests.get(self.address, params=params)
        return


