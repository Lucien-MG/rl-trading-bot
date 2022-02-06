#!/usr/bin/python3
#‑∗‑ coding: utf‑8 ‑∗‑

from core.log.log_interface import LogInterface

import torch.utils.tensorboard as tensorboard

class TensorboardLog(LogInterface):

    def start(self):
        self.writer = tensorboard.SummaryWriter()

    def stop(self,):
        self.writer.close()

    def log(self, name, value, step):
        self.writer.add_scalar(name, value, step)

Logger = TensorboardLog
