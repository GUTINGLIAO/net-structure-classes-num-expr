import time


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())