from icecream import ic

from multiprocessing import Process, Queue ,Lock, Manager
import multiprocessing
import cv2 as cv
import numpy as np
import time
import starlette
# import nicegui.globals    
from nicegui import ui
# from queue import Queue


    



cap = cv.VideoCapture("/home/infinus/IDP_M213/2022-11-11 16-00-12.mp4")
test = 100
# q = Manager().Queue()


ic("running")
q = Manager().Queue()
ic(q)

def cvStart():
    global cvProcess 
    try:
        cvProcess
    except NameError:
        cvProcess = None

    if cvProcess is None:
        ic("Starting CV")
        cvProcess = Process(target=cvServe, args = (q,))
        cvProcess.start()
    else:
        ui.notify("Already Running")

def cvStop():
    cvProcess.terminate()
    cvProcess.join()
    ic("Exited CV")
    
def cvServe():
    ic("Spawned CV")
    ic(q)
    
    while True:
        ret, frame = cap.read()
        # ic(frame)
        if not ret:
            raise Exception("Can't receive frame (stream end?). Exiting ...")
        cv.imshow("test", frame)
        _, frame = cv.imencode('.jpg', frame)
        q.put(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')
        
        # ic(q.qsize())
        time.sleep(0.1)
        if cv.waitKey(1) == ord('q'):
            break

  
@ui.get("/1.mjpeg")
async def videoStream():
    ic(q)
    ic(test)
    data = await q.get()
    ic(data)
    return starlette.responses.PlainTextResponse(data)

ui.label('Hello NiceGUI!')
ui.button('Spawn CV', on_click=cvStart)
ui.button('Kill CV', on_click=cvStop)


# ui.image(source='localhost:')

ui.run()

# protect the entry point
if __name__ == '__main__':
    multiprocessing.set_start_method("fork")
    multiprocessing.freeze_support()