import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np


if __name__ == '__main__':
    test_data = datasets.MNIST(
        root='./dataset/',
        train=False,
        transform=transforms.ToTensor()
    )

    # model = torch.load('./MSE/checkpoint_29.pkl')
    # model = torch.load('./L1/checkpoint_29.pkl')
    # model = torch.load('./SmoothL1/checkpoint_29.pkl')
    model = torch.load('./CNNBase/checkpoint_29.pkl')

    view_data = test_data.data[:200]

    # view_data = view_data.view(-1, 28 * 28).type(torch.FloatTensor) / 255. # Autoencoder用
    view_data = view_data.type(torch.FloatTensor).unsqueeze(1) / 255.   # AutoencoderCNN用

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    encoded, decoded = model(view_data.to(device))

    fig = plt.figure("Image")
    ax = Axes3D(fig)

    X = encoded.data[:, 0].cpu().numpy()
    Y = encoded.data[:, 1].cpu().numpy()
    Z = encoded.data[:, 2].cpu().numpy()

    values = test_data.targets[:200].numpy()
    for x, y, z, s in zip(X, Y, Z, values):
        c = cm.rainbow(int(255 * s / 9))
        ax.text(x, y, z, s, backgroundcolor=c)

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())

    plt.show()

    with torch.no_grad():
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(10):
            a[0][i].imshow(test_data.data[i])
            a[1][i].imshow(np.reshape(decoded[i].cpu(), (28, 28)))
        plt.show()

