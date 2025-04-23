import torch
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from custom_data import CustomYoloDataset
from loss import SumSquaredErrorLoss
from models import *

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.autograd.set_detect_anomaly(True)
    now = datetime.now()

    model = YOLOv1().to(device)
    loss_function = SumSquaredErrorLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE
    )

    # Dataset directory
    dir = r"C:\Users\alkan\.cache\kagglehub\datasets\a2015003713\militaryaircraftdetectiondataset\versions\87\dataset"
    train_set = CustomYoloDataset("train", dir, dir, normalize=True, augment=True)
    test_set = CustomYoloDataset("test", dir, dir, normalize=True, augment=True)

    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        num_workers=8,
        persistent_workers=True,
        drop_last=True,
        shuffle=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.BATCH_SIZE,
        num_workers=8,
        persistent_workers=True,
        drop_last=True
    )

    # Output folder
    root = os.path.join('models', 'yolo_v1', now.strftime('%m_%d_%Y'), now.strftime('%H_%M_%S'))
    weight_dir = os.path.join(root, 'weights')
    if not os.path.isdir(weight_dir):
        os.makedirs(weight_dir)

    # Metrics
    train_losses = np.empty((2, 0))
    test_losses = np.empty((2, 0))
    train_errors = np.empty((2, 0))
    test_errors = np.empty((2, 0))

    def save_metrics():
        np.save(os.path.join(root, 'train_losses'), train_losses)
        np.save(os.path.join(root, 'test_losses'), test_losses)
        np.save(os.path.join(root, 'train_errors'), train_errors)
        np.save(os.path.join(root, 'test_errors'), test_errors)

    #####################
    #       Train       #
    #####################
    for epoch in tqdm(range(config.WARMUP_EPOCHS + config.EPOCHS), desc='Epoch'):
        model.train()
        train_loss = 0
        for data, labels, _ in tqdm(train_loader, desc='Train', leave=False):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = model.forward(data)
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() / len(train_loader)
            del data, labels

        train_losses = np.append(train_losses, [[epoch], [train_loss]], axis=1)
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")

        if epoch % 4 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = 0
                for data, labels, _ in tqdm(test_loader, desc='Test', leave=False):
                    data = data.to(device)
                    labels = labels.to(device)

                    predictions = model.forward(data)
                    loss = loss_function(predictions, labels)

                    test_loss += loss.item() / len(test_loader)
                    del data, labels
            test_losses = np.append(test_losses, [[epoch], [test_loss]], axis=1)
            print(f"[Epoch {epoch}] Test Loss: {test_loss:.4f}")
            save_metrics()

    save_metrics()
    torch.save(model.state_dict(), os.path.join(weight_dir, 'final'))