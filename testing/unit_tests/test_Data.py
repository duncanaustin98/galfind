
import pytest
import os

os.environ["GALFIND_CONFIG_DIR"] = os.getcwd()
os.environ["GALFIND_CONFIG_NAME"] = "test_galfind_config.ini"

import galfind