import time
from functools import wraps
def timing(func):
    """This decorator prints the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'Elapsed time: {end - start} seconds')
        return result

    return wrapper