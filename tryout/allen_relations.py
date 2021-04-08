
from datetime import datetime

class TemporalInterval(object):
    def __init__(self) -> None:
        self.__start = datetime(1900, 1, 1, 0, 0)
        self.__end = datetime(2100, 1, 1, 0, 0)
        self.__duration = self.__end - self.__start

    def set_start(self, start, fix_end=True):
        self.__start = start
        if fix_end:
            self.__duration = self.__end - self.__start
        else:
            self.__end = self.__start + self.__duration

    def set_end(self, end, fix_start=True):
        self.__end = end
        if fix_start:
            self.__duration = self.__end - self.__start
        else:
            self.__start = self.__end - self.__duration

    def set_duration(self, duration, fix_start=True):
        self.__duration = duration
        if fix_start:
            self.__end = self.__start + self.__duration
        else:
            self.__start = self.__end - self.__duration

    def precedes(self, an_interval) -> bool:
        return self.__end < an_interval.__start

    def preceded_by(self, an_interval) -> bool:
        return an_interval.precedes(self)

    def meets(self, an_interval) -> bool:
        return self.__end == an_interval.__start

    def met_by(self, an_interval) -> bool:
        return an_interval.meets(self)

    def overlaps(self, an_interval) -> bool:
        return (
                self.__start < an_interval.__start < self.__end < an_interval.__end
        )

    def overlaped_by(self, an_interval) -> bool:
        return an_interval.overlaps(self)

    def finishes(self, an_interval) -> bool:
        return (
                self.__end == an_interval.__end and
                self.__start > an_interval.__start
        )

    def finished_by(self, an_interval) -> bool:
        return an_interval.finishes(self)

    def during(self, an_interval) -> bool:
        return (
                an_interval.__start < self.__start and
                self.__end < an_interval.__end
        )

    def contains(self, an_interval) -> bool:
        return an_interval.during(self)

    def starts(self, an_interval) -> bool:
        return (
                self.__start == an_interval.__start and
                self.__end < an_interval.__end
        )

    def started_by(self, an_interval) -> bool:
        return an_interval.starts(self)

    def equals(self, an_interval) -> bool:
        return (
                an_interval.__start == self.__start and
                an_interval.__end == self.__end
        )