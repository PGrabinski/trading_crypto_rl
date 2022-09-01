# common library
import pandas as pd
import numpy as np
import time

from stable_baselines3.common.vec_env import DummyVecEnv

# preprocessor
from preprocessing.preprocessors import *
# config
from config.config import *
# model
from model.models import *
import os

BEGINING_DATE = 2020600
END_TRAIN_DATE = 20210900

BEGINING_TEST_DATE = 20210901
END_TEST_DATE = 20220201


rebalance_window = 30
validation_window = 30

def run_model() -> None:
    """Train the model."""
    # read and preprocess data
    preprocessed_path = "done_data.csv"
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)
    else:
        data = preprocess_data()
        data = add_turbulence(data)
        data.to_csv(preprocessed_path)

    # print(data.head())
    # print(data.size)

    # 2015/10/01 is the date that validation starts
    # 2016/01/01 is the date that real trading starts
    # unique_trade_date needs to start from 2015/10/01 for validation purpose
    unique_trade_date = data[(data.datadate > BEGINING_TEST_DATE)&(data.datadate <= END_TEST_DATE)].datadate.unique()
    # print(unique_trade_date)

    # rebalance_window is the number of months to retrain the model
    # validation_window is the number of months to validation the model and select for trading
    
    ## Ensemble Strategy
    run_ensemble_strategy(df=data, 
                          unique_trade_date= unique_trade_date,
                          rebalance_window = rebalance_window,
                          validation_window=validation_window,
                          begining_date=BEGINING_DATE,
                          end_train_date=END_TRAIN_DATE)

    #_logger.info(f"saving model version: {_version}")

if __name__ == "__main__":
    run_model()
