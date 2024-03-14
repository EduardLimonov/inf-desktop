from typing import Callable

from logs.logs import logger


def core_callback_secure(fn: Callable):
    def wrapped(*args, **kwargs):
        try:
            logger.debug(f"Try to execute network-dependent function {fn.__name__}")
            response = fn(*args, **kwargs)
        except Exception as e:
            logger.exception(e)
            response = None
        return response

    return wrapped
