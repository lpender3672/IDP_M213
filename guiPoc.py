from icecream import ic
from nicegui import ui
from multiprocessing import Process, Queue, Lock
import guiPocStream

import time
import starlette

# if __name__ == "__main__":
ui.label('Hello NiceGUI!')
ui.button('Spawn CV', on_click=guiPocStream.main)


ui.image(source="http://localhost:8081/stream/video.mjpeg")

ui.run()