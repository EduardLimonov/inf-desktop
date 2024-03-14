import datetime
from typing import List, Sequence

import numpy as np
from PyQt5.QtWidgets import QProgressBar


class CustomPB:
    iterable: List
    bar: QProgressBar
    text: str

    def __init__(self, iterable: Sequence, pbar: QProgressBar, text: str = ""):
        self.iterable = list(iterable)
        self.text = text
        self.bar = pbar
        pbar.setVisible(True)
        pbar.setValue(0)
        pbar.setFormat(f"{text} (%p%)")

    @staticmethod
    def __format_tdelta(tdelta) -> str:
        if type(tdelta) is int:
            return "?"
        else:
            mins = tdelta.seconds // 60
            seconds = tdelta.seconds % 60
            hours = mins // 60
            mins %= 60
            if hours == 0:
                return f"{str(mins).zfill(2)}:{str(seconds).zfill(2)}"
            else:
                return f"{str(hours).zfill(2)}:{str(mins).zfill(2)}:{str(seconds).zfill(2)}"

    def __iter__(self):
        tdeltas = []
        start_t = datetime.datetime.now()
        prev_t = start_t

        for cnt in range(len(self.iterable)):
            mean_tdelta = 0
            if cnt > 0:
                next_t = datetime.datetime.now()
                tdeltas.append(next_t - prev_t)
                mean_tdelta = np.mean(tdeltas)
                prev_t = next_t

            current_prog = cnt / len(self.iterable)
            # self.bar.progress(
            #     current_prog,
            #     text=f"{self.text}: {int(current_prog * 100)}% {cnt} / {len(self.iterable)} "
            #     f"(осталось ~ {self.__format_tdelta(mean_tdelta * (len(self.iterable) - cnt))})",
            # )
            self.bar.setValue(int(current_prog * 100))
            self.bar.setFormat(f"{self.text} осталось ~ "
                               f"{self.__format_tdelta(mean_tdelta * (len(self.iterable) - cnt))}"
                               f" {cnt + 1} / {len(self.iterable)} (%p%) ")

            yield self.iterable[cnt]

        self.bar.hide()

