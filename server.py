from multiprocessing.managers import SyncManager
from multiprocessing import Queue


class MyManager(SyncManager):
    pass

buf = Queue()

def get_dict():
    return buf

if __name__ == "__main__":
    MyManager.register("syncdict", callable=lambda: buf)
    manager = MyManager(("127.0.0.1", 5000), authkey=b"password")
    manager.start()
    input()
    manager.shutdown()