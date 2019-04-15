# import atexit
# from time import time, strftime, localtime
# from datetime import timedelta

from contextlib import ContextDecorator
from timeit import default_timer as timer

class Tim(ContextDecorator):
    """ Class to log timeing of block

    Can be used as decorator or as context.

    Usage as decorator:
    @Tim("My Timer", "timeout")
    def foo():
        do_stuff
    
    Usage as context:
    def foo():
        with Tim("My Timer", "timeout"):
           do_stuff
    """

    def __init__(self, name, filename):
        self.start = 0
        self.end = 0
        self.elapsed = 0
        self.name = name
        self.filename = filename

    def __enter__(self):
        self.start = timer()

    def __exit__(self, *args):
        self.end = timer()
        self.elapsed = self.end - self.start
        with open(self.filename, 'a') as f:
            f.write("%s, %s, %s, %s\n" % (self.name, self.start, self.end, self.elapsed))

