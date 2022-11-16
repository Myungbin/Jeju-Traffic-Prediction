import torch
import torch.nn as nn
import torch.optim as optim

import random
import optuna
import pandas as pd

from optuna.trial import TrialState

from torch.utils.data import DataLoader, Dataset, random_split
from feature import make_dataset
from tqdm import tqdm

class CFG:
    device = 'cuda'
    batch_size = 2048

    train_path = './data/data_parquet/train.csv'
    test_path = './data/data_parquet/test.csv'
    submission_path = './sample_submission.csv'
    
    weights_path = './weights/'

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(38, 512)
        self.layer1 = self.make_layers(512, num_repeat=30)
        self.fc5 = nn.Linear(512, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.fc5(x)
        return x

    def make_layers(self, value, num_repeat):
        layers = []
        for _ in range(num_repeat):
            layers.append(nn.Linear(value, value))
            layers.append(nn.ReLU(inplace=True))
            # layers.append(nn.Dropout(0.1))

        return nn.Sequential(*layers)


class tablurDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values).to(CFG.device, dtype=torch.float)
        self.y = torch.tensor(y.values).to(CFG.device, dtype=torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(random_seed)

def train_one_epoch(model, optimizer, dataloader, criterion, epoch):
    model.train()
    dataset_size = 0
    running_loss = 0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for step, data in bar:
        X = data[0].to(CFG.device, dtype=torch.float)
        Y = data[1].to(CFG.device, dtype=torch.float)
        outputs = model(X)
        loss = criterion(outputs, Y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()*CFG.batch_size
        dataset_size += CFG.batch_size
        epoch_loss = running_loss/dataset_size

        bar.set_postfix(EPOCH=epoch, TRAIN_LOSS=epoch_loss)
    print(f'train_loss:{epoch_loss}')

def val_one_epoch(model, dataloader, criterion, epoch, val_loss):
    model.eval()
    with torch.no_grad():
        dataset_size = 0
        running_loss = 0
        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        
        for step, data in bar:
            X = data[0].to(CFG.device, dtype=torch.float)
            Y = data[1].to(CFG.device, dtype=torch.float)
            outputs = model(X)
            loss = criterion(outputs, Y)
            
            running_loss += loss.item()*CFG.batch_size
            dataset_size += CFG.batch_size
            epoch_loss = running_loss/dataset_size
            
            val_loss.append(epoch_loss)
            bar.set_postfix(EPOCH=epoch, VAL_LOSS=epoch_loss)
            
        print(f'val_loss:{epoch_loss}')

def objective(trial):
    set_seed(42)
    model = NeuralNet().to(CFG.device)
    criterion = nn.L1Loss()
    
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    epochs = trial.suggest_int("epochs", 5, 100)

    all_dataset = tablurDataset(train_X, train_Y)
    train_dataset, val_dataset = random_split(all_dataset, [int(len(train_X)*0.8),len(train_X)-int(len(train_X)*0.8)])

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size)
    
    for epoch in range(epochs):
        val_loss = []
        train_one_epoch(model, optimizer, train_loader, criterion, epoch)
        val_one_epoch(model, val_loader, criterion, epoch, val_loss)
        mean_val_loss = sum(val_loss)/len(val_loss)
        trial.report(mean_val_loss, epoch)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return mean_val_loss
    
if __name__ == '__main__':
    train_X, train_Y, test_X = make_dataset(CFG.train_path, CFG.test_path)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))