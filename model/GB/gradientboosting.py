import torch
import torch.nn as nn
import torch.optim as optim

import random
import optuna
import pandas as pd

from second_th_dataset.dataset import make_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error

class CFG:
    device = 'cuda'
    batch_size = 2048

    train_new_path = './data/data_parquet/train_new.parquet'
    test_new_path = './data/data_parquet/test_new.parquet'
    holiday_path = './data/data_csv/국가공휴일.csv'
    submission_path = './data/data_csv/sample_submission.csv'
    
    
def readnewData():
    global X, y, x_test, sample_submission
    X = pd.read_parquet(CFG.train_new_path)
    y = pd.DataFrame()
    y['target'] = X['target'].copy()
    X = X.drop('target', axis=1)

    x_test = pd.read_parquet(CFG.test_new_path)
    sample_submission = pd.read_csv(CFG.submission_path)

 
def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.08)
    n_estimators =  trial.suggest_int('n_estimators', 100, 500)
    max_depth = trial.suggest_int('max_depth', 2, 10)
    
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 10, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 5, 20)
    
    
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(n_estimators = n_estimators,
                                    learning_rate = learning_rate,
                                    loss = 'absolute_error',
                                    max_depth = max_depth,
                                    max_features='sqrt',
                                    min_samples_leaf=min_samples_leaf,
                                    min_samples_split=min_samples_split,
                                    random_state = 42
                                  )
    
    print('model fit start')
    model.fit(x_train, y_train.values.ravel())
    print('model fit end')
    rf_pred = model.predict(x_valid)
    score = mean_absolute_error(y_valid, rf_pred)
    print(f'score : {score}')
    
    return score
    
    
    
if __name__ == '__main__':
    
    readnewData()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
