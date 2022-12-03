from icecream import ic

from multiprocessing import Process, Queue ,Lock, Manager
from multiprocessing.managers import SyncManager
from typing import Optional, Dict, Any, Union
import multiprocessing
import cv2 as cv
import numpy as np
import time
import starlette
import base64
# import nicegui.globals    
from nicegui import ui

# from queue import Queue



class MyManager(SyncManager):
    ...



# cap = cv.VideoCapture("/home/infinus/IDP_M213/2022-11-11 16-00-12.mp4")
# test = 100
# q = Manager().Queue()

def cvStart():
    global cvProcess 
    try:
        cvProcess
    except NameError:
        cvProcess = None

    if cvProcess is None:
        ic("Starting CV")
        cvProcess = Process(target=cvServe)
        cvProcess.start()
    else:
        ui.notify("Already Running")

def cvStop():
    cvProcess.terminate()
    cvProcess.join()
    ic("Exited CV")
    
def cvServe():
    manager = MyManager(("127.0.0.1", 5000), authkey=b"password")
    manager.connect()
    manager.register("syncdict")
    q = manager.syncdict()
    ic("Spawned CV")
    # ic(q)
    cap = cv.VideoCapture("/home/infinus/IDP_M213/2022-11-11 16-00-12.mp4")
    while True:
        ret, frame = cap.read()
        # frame = frame[0:100, 0:100]
        # ic(q)
        if not ret:
            raise Exception("Can't receive frame (stream end?). Exiting ...")
        # cv.imshow("test", frame)
        _, frame = cv.imencode('.jpg', frame)
        # q.put(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')
        dat = base64.b64encode(frame).decode('utf-8')
        # ic(dat)
        q.put(dat)
        
        # ic(q.qsize())
        time.sleep(0.1)
        if cv.waitKey(1) == ord('q'):
            break

# async def grabFrame():
#     manager = MyManager(("127.0.0.1", 5000), authkey=b"password")
#     manager.connect()
#     manager.register("syncdict")
#     q = manager.syncdict()
#     while q.qsize() > 0:
#         yield q.get(False)
#     yield None
  
# @ui.get("/1.mjpeg")
# async def videoStream():
#     # manager = MyManager(("127.0.0.1", 5000), authkey=b"password")
#     # manager.connect()
#     # manager.register("syncdict")
#     # q = manager.syncdict()
#     # q = Manager().Queue()
#     # ic(q.qlength())
#     # ic(test)
#     # data = q.get(False)
#     header = {
#         "Age": "0",
#         'Cache-Control': 'no-cache, private',
#         'Pragma': 'no-cache',
#         "Content-Type": 'multipart/x-mixed-replace; boundary=frame'

#     }
#     # ic(data)
#     frame = grabFrame()
#     return starlette.responses.StreamingResponse(frame, status_code = 206, headers = header)


manager = MyManager(("127.0.0.1", 5000), authkey=b"password")
manager.connect()
manager.register("syncdict")
q = manager.syncdict()

ui.label('Hello NiceGUI!')
ui.button('Spawn CV', on_click=cvStart)
ui.button('Kill CV', on_click=cvStop)


img = ui.html()
def refresh():
    if(q.qsize()) == 0:
        return
    img.content = "<img src='data:image/jpeg;base64," + q.get() +"'>"
ui.timer(interval = 0.1, callback=refresh)

ui.run()
