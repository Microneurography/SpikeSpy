from PySide6.QtCore import QTimer


def qsignal_throttle(func, interval=33):
    # connect to signals
    throttleSignalTimer = QTimer()
    throttleSignalTimer.setInterval(interval)  # 30 fps
    throttleSignalTimer.timeout.connect(func)
    throttleSignalTimer.setSingleShot(True)

    def start(*args, **kwargs):
        if not throttleSignalTimer.isActive():
            throttleSignalTimer.start()

    return start


from PySide6.QtCore import QTimer
from functools import wraps


def qsignal_throttle2(func, interval=33):
    """Throttle a function using a QTimer."""

    def wrapper(*args, **kwargs):
        # Ensure each instance gets its own QTimer
        if not hasattr(wrapper, "_throttle_timer"):
            wrapper._throttle_timer = QTimer()
            wrapper._throttle_timer.setInterval(interval)
            wrapper._throttle_timer.setSingleShot(True)
            wrapper._throttle_timer.timeout.connect(lambda: func(*args, **kwargs))
            func(*args, **kwargs)  # Call the function immediately

        elif not wrapper._throttle_timer.isActive():
            # func(*args, **kwargs)  # Call the function immediately
            wrapper._throttle_timer.start()

    return wrapper


def qsignal_throttle_wrapper(interval=33):
    """Decorator to throttle a function using qsignal_throttle."""

    def decorator(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            # Create a new throttled version of the function for each instance
            if not hasattr(wrapped_func, "_throttle_func"):
                print("new timer")
                wrapped_func._throttle_func = qsignal_throttle2(func, interval)
            return wrapped_func._throttle_func(*args, **kwargs)

        return wrapped_func

    return decorator
