#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:43:05 2023

@author: austind
"""

# decorators.py
import os
import time
import warnings

import yagmail
from astropy import units as u

from . import galfind_logger


def run_in_dir(path):
    """Decorator factory that runs the wrapped function inside a given directory.

    Changes the current working directory to `path` (creating it first if
    it does not already exist) before calling the wrapped function, and
    restores the original working directory afterwards.

    Parameters
    ----------
    path : `str`
        Path to the directory to change into before calling the wrapped
        function.

    Returns
    -------
    `callable`
        Decorator that wraps a function to be run inside `path`.
    """
    def decorated(func):
        def wrapper(*args, **kwargs):
            cwd = os.getcwd()
            if not os.path.exists(path):
                os.makedirs(path)
            os.chdir(path)
            #print(f"Changed directory to {path}")
            return_value = func(*args, **kwargs)
            os.chdir(cwd)
            #print(f"Changed directory back to {cwd}")
            return return_value
        return wrapper
    return decorated

def run_in_self_dir(get_dir):
    """Decorator factory that runs a bound method inside a `self`-dependent directory.

    Like `run_in_dir`, but for instance methods: the directory to run
    inside is obtained by calling `get_dir(self)` at call time (rather than
    being fixed at decoration time), created if it does not already exist,
    and the original working directory is restored afterwards.

    Parameters
    ----------
    get_dir : `callable`
        Function taking the instance (`self`) and returning the `str`
        directory path to run the wrapped method inside.

    Returns
    -------
    `callable`
        Decorator that wraps an instance method to be run inside the
        directory returned by `get_dir(self)`.
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            dir = get_dir(self)  # Access self attribute at call time
            cwd = os.getcwd()
            if not os.path.exists(dir):
                os.makedirs(dir)
            os.chdir(dir)
            return_value = func(self, *args, **kwargs)
            os.chdir(cwd)
            return return_value
        return wrapper
    return decorator


def log_time(logging_level, out_unit: u.Quantity = u.hour):
    """Decorator factory that logs the execution time of the wrapped function.

    Logs an INFO-level message before the wrapped function is called, and
    a message at `logging_level` reporting the elapsed execution time
    (converted to `out_unit`) once it returns.

    Parameters
    ----------
    logging_level : `int`
        Logging level (e.g. `logging.INFO`) to log the elapsed execution
        time at.
    out_unit : `astropy.units.Quantity`, optional
        Unit to convert the measured execution time into before logging.
        Default is `astropy.units.hour`.

    Returns
    -------
    `callable`
        Decorator that wraps a function to log its execution time.
    """
    def decorated(func):
        def wrapper(*args, **kwargs):
            galfind_logger.info(
                f"Running {func.__name__}!"
            )
            start_time = time.time()
            return_value = func(*args, **kwargs)
            end_time = time.time()
            # log at required level
            galfind_logger.log(
                logging_level,
                f"{func.__name__} executed in {((end_time - start_time) * u.s).to(out_unit):.1f}!"
            )
            return return_value

        return wrapper

    return decorated


def hour_timer(func):
    """Decorator that prints the execution time of the wrapped function, in hours.

    Parameters
    ----------
    func : `callable`
        Function to time.

    Returns
    -------
    `callable`
        Wrapped version of `func` that prints its execution time (converted
        to hours) to stdout after it returns.
    """
    def wrapper(*args, **kwargs):
        t1 = time.time()
        return_value = func(*args, **kwargs)
        t2 = time.time()
        print(
            f"Function {func.__name__!r} executed in {((t2-t1) * u.s).to(u.h)}"
        )
        return return_value

    return wrapper


def ignore_warnings(func):
    """Decorator that suppresses all warnings raised while the wrapped function runs.

    Parameters
    ----------
    func : `callable`
        Function to run with warnings suppressed.

    Returns
    -------
    `callable`
        Wrapped version of `func` that runs inside a
        `warnings.catch_warnings()` context with all warnings ignored.
    """
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper


# should also include the 'logged' output here!
# REMEMBER TO PUT IN .gitignore
def email_update(
    to="duncan.austin@postgrad.manchester.ac.uk",
    send_start=False,
    send_end=True,
):
    """Decorator factory that emails status updates about the wrapped function.

    Uses `yagmail` (with a hardcoded OAuth2 credential file) to send an
    email when the wrapped function starts, an email when it ends
    successfully, and a termination email (before re-raising) if it raises
    an exception.

    Parameters
    ----------
    to : `str`, optional
        Email address to send the update(s) to. Default is
        `"duncan.austin@postgrad.manchester.ac.uk"`.
    send_start : `bool`, optional
        Intended to control whether a start email is sent; not currently
        read by the wrapped function, which always sends the start email.
        Default is `False`.
    send_end : `bool`, optional
        Intended to control whether an end email is sent; not currently
        read by the wrapped function, which always sends the end email on
        success. Default is `True`.

    Returns
    -------
    `callable`
        Decorator that wraps a function to send email status updates.
    """
    def decorated(func):
        def wrapper(*args, **kwargs):
            # setup gmail
            setup = yagmail.SMTP(
                "tcharvey303",
                oauth2_file="/nvme/scratch/work/tharvey/scripts/testing/client_secret_228822080160-3n68iam26fj8pf8mjmcamfse7gjb12ks.apps.googleusercontent.com.json",
            )
            # compose starting email
            setup.send(
                to,
                f"Morgan START: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}",
                f"Starting {func.__name__}",
            )
            # try to run decorated function
            try:
                return_value = func(*args, **kwargs)
            except:
                # compose failure email
                setup.send(
                    f"Morgan TERMINATE: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}",
                    f"Terminating {func.__name__}",
                )
                raise (Exception(f"Terminating {func.__name__}"))
            # compose ending email
            setup.send(
                to,
                f"Morgan END: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}",
                f"Ending {func.__name__}",
            )
            return return_value

        return wrapper

    return decorated


# Parallelization decorators
def n_cores(n):
    """Placeholder decorator factory intended to run the wrapped function using `n` cores.

    As currently implemented, the core/thread allocation logic is not yet
    filled in, so the wrapped function is simply called unchanged.

    Parameters
    ----------
    n : `int`
        Intended number of cores to run the wrapped function with.

    Returns
    -------
    `callable`
        Decorator that wraps a function, currently calling it unmodified.
    """
    def decorated(func):
        def wrapper(*args, **kwargs):
            # setup how many cores to use here with argument 'n'
            return_value = func(*args, **kwargs)
            # revert to default number of cores
            return return_value

        return wrapper

    return decorated
