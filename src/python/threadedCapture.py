import cv2, queue, threading, time

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 760)
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1012)  
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()

#this section only for testing

if __name__ == "__main__":
  cap = VideoCapture("http://localhost:8081/stream/video.mjpeg")
  while True:
    time.sleep(.5)   # simulate time between events
    frame = cap.read()
    cv2.imshow("frame", frame)
    if chr(cv2.waitKey(1)&255) == 'q':
      break
