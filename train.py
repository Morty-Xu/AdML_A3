import torch
from model import Autoencoder, AutoencoderCNN
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loss import calLoss
from tqdm import tqdm, trange
import os


if __name__ == '__main__':
    BATCH_SIZE = 128

    # load data
    train_data = datasets.MNIST(root='./dataset/', train=True, transform=transforms.ToTensor(), download=True)
    test_data = datasets.MNIST(root='./dataset/', train=False, transform=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

    # model prepare
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    LR = 1e-4

    # model = Autoencoder()
    model = AutoencoderCNN()

    model = model.to(device)
    K = model.parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loss_func = nn.MSELoss()
    # loss_func = nn.L1Loss()
    # loss_func = nn.SmoothL1Loss()

    writer = SummaryWriter('./logs/')

    EPOCH = 30

    # 训练集loss列表
    train_loss_list = []
    # 测试集loss列表
    test_loss_list = []
    # 创建checkpoints文件夹
    os.mkdir('./checkpoints')
    # 模型训练
    for epoch in trange(EPOCH):
        epoch_iterator = tqdm(train_loader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            data, labels = batch

            # 由于是无监督模型，只采用data部分，并且复制一份用于计算loss
            # x = y = data.view(-1, 28 * 28)  # Autoencoder用
            x = y = data
            encoded, decoded = model(x.to(device))
            loss = loss_func(decoded, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 计算训练集loss
        train_loss = calLoss(train_loader, model, loss_func, device)
        # 计算测试集loss
        test_loss = calLoss(test_loader, model, loss_func, device)
        # 存储loss
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        # 绘制图像
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('test_loss', test_loss, epoch)
        # 保存本次训练模型

        torch.save(model, './checkpoints/checkpoint_{}.pkl'.format(epoch))


