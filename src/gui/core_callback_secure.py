import logging


def core_callback_secure(fn):
    def wrapped(*args, **kwargs):
        try:
            response = fn(*args, **kwargs)
        except Exception as e:
            logging.exception(e)
            response = None
        return response

    return wrapped
