import functools
import threading

def watcher(watch_fn):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            def _watch():
                try:
                    watch_fn(
                        args=args,
                        kwargs=kwargs,
                        result=result,
                    )
                except Exception:
                    pass

            threading.Thread(target=_watch, daemon=True).start()
            return result

        return wrapper
    return decorator
