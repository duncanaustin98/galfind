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
from astropy.units import u


def run_in_dir(path):
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


def hour_timer(func):
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
    def decorated(func):
        def wrapper(*args, **kwargs):
            # setup how many cores to use here with argument 'n'
            return_value = func(*args, **kwargs)
            # revert to default number of cores
            return return_value

        return wrapper

    return decorated
