#!/usr/bin/python3
# ‑∗‑ coding: utf‑8 ‑∗‑

import csv
import time

from queue import Queue
from threading import Thread

class Tracer:

    def __init__(self, path, fields, cooldown=3):
        self.path = path
        self.fields = fields
        self.cooldown = cooldown
        self.queue = Queue()

        self.run = False
        self.thread = Thread(target = self.__registering_loop__)

    def __flush__(self):
        with open(self.path, 'a') as csvfile:
            csv_writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=self.fields)

            while not self.queue.empty():
                element = self.queue.get()
                csv_writer.writerow(element)

    def __registering_loop__(self):
        while self.run:
            self.__flush__()
            time.sleep(self.cooldown)

    def register(self, values: list):
        self.queue.put(values)

    def open(self):
        self.run = True
        self.thread.start()

    def close(self):
        self.run = False
        self.thread.join()
        self.__flush__()
