
import sys
import os
import warnings
from erfa import ErfaWarning

def suppress_output(func, *args, **kwargs):
    # Save the original stdout
    original_stdout = sys.stdout
    try:
        # Suppress stdout by redirecting to devnull
        sys.stdout = open(os.devnull, 'w')
        
        # Suppress specific warnings (ErfaWarning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ErfaWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            result = func(*args, **kwargs)
    finally:
        # Always restore stdout, even if an error occurs
        sys.stdout = original_stdout
    return result
