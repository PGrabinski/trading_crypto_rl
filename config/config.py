import pathlib

import datetime
import os

TRAINING_DATA_FILE = './data/crypto_only_full.csv'

now = str(datetime.datetime.now()).replace(' ', '_').replace(':', '_').split('.')[0]
TRAINED_MODEL_DIR = f"trained_models/{now}"

os.makedirs(TRAINED_MODEL_DIR)
TESTING_DATA_FILE = "test.csv"