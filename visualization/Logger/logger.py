import numpy as np
import matplotlib.pyplot as plt

with open('logger.txt') as f:
    lines = f.readlines()

    iteration = [int(line.split()[1]) for line in lines]
    epoch = [int(line.split()[4]) for line in lines]
    train_loss = [float(line.split()[8]) for line in lines]
    val_loss = [float(line.split()[12]) for line in lines]
    DSC = [float(line.split()[-1]) for line in lines]

    max_epoch = max(epoch)
    max_iteration = max(iteration)

    num = epoch.count(max_epoch)

    epochs = epoch = [i/num for i in range(max_iteration)]

    plt.plot(epoch, train_loss,'b', label='train loss')
    plt.plot(epoch, val_loss,'r', label='val loss')

    plt.title('Train and Val Loss')
    plt.legend(['Train Loss', 'Val Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.savefig('Loss.png', dpi=300)

    plt.figure()

    plt.title('DSC')
    plt.plot(epoch, DSC)
    plt.xlabel('Epoch')
    plt.ylabel('DSC')
    plt.legend(['DSC'])

    plt.savefig('DSC.png', dpi=300)

