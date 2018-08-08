import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os

class Watcher:
    DIRECTORY_TO_WATCH = "data"

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Error")

        self.observer.join()


class Handler(FileSystemEventHandler):

    counter = 0
    imglist = list()
    @staticmethod
    def on_any_event(event):

        if event.is_directory:
            return None

        elif event.event_type == 'created':
            # Take any action here when a file is first created.
            print("Received created event - %s." % event.src_path)
            Handler.counter = Handler.counter + 1
            #Handler.imglist.append(event.src_path)
            #print(event.src_path)
            if Handler.counter >= 10:
                #print('ten images')
                Handler.counter = 0
                #os.system('./run_preprocess.sh')

        '''elif event.event_type == 'modified':
            # Taken any action here when a file is modified.
            print("Received modified event - %s." % event.src_path)'''


if __name__ == '__main__':
    w = Watcher()
    w.run()