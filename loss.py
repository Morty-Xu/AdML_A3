import torch
from tqdm import tqdm


def calLoss(data_loader, model, loss_func, device):
    loss = 0
    with torch.no_grad():
        for data, labels in tqdm(data_loader):

            # x = y = data.view(-1, 28 * 28) # Autoencoder用
            x = y = data
            encoded, decoded = model(x.to(device))
            loss = loss_func(decoded, y.to(device))

    loss = loss.item() / len(data_loader)

    return loss

