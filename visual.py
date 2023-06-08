import numpy as np
import matplotlib.pyplot as plt
import os


def plot_interval(arr, total_epochs):
    for layer, data in arr.items():
        if not os.path.exists('./fig/interval'):
            os.makedirs('./fig/interval')
        data = np.reshape(data, (total_epochs, -1))
        n = len(data)
        x = np.arange(1, n+1)
        y = np.mean(data, axis=1)
        ymin = np.min(data, axis=1)
        ymax = np.max(data, axis=1)

        plt.plot(x, y, color='blue')
        plt.fill_between(x, ymin, ymax, color='lightblue')

        plt.xlabel('Data Point')
        plt.ylabel('Value')
        plt.title('Interval Plot')

        plt.savefig('./fig/interval/{}.png'.format(layer))
        plt.clf()