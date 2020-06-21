import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def cullenfrey(xd, title):
    plt.figure(num=None, figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
    fig, ax = plt.subplots()
    i = 0
    for i in range(len(xd)):
        if i == 0:
            cor = "lightgray"
        if i == 1:
            cor = "orange"
        if i == 2:
            cor = "green"
        if i == 3:
            cor = "red"
        if i == 4:
            cor = "yellow"
        maior = np.max(xd[i][1])
        polyX1 = maior if maior > 4.4 else 4.4
        polyY1 = polyX1 + 1
        polyY2 = 3 / 2. * polyX1 + 3
        y_lim = polyY2 if polyY2 > 10 else 10

        x = [0, polyX1, polyX1, 0]
        y = [1, polyY1, polyY2, 3]
        scale = 1
        poly = Polygon(np.c_[x, y] * scale, facecolor='#1B9AAA', edgecolor='#1B9AAA', alpha=0.5)
        ax.add_patch(poly)
        plt.plot(float(xd[i][1]) ** 2, float(xd[i][2]) + 3, marker="o", c=cor, label=xd[i][0])
    ax.plot(0, 4.187999875999753, label="logistic", marker='+', c='black')
    ax.plot(0, 1.7962675925351856, label="uniform", marker='^', c='black')
    ax.plot(4, 9, label="exponential", marker='s', c='black')
    ax.plot(0, 3, label="normal", marker='*', c='black')
    ax.plot(np.arange(0, polyX1, 0.1), 3 / 2. * np.arange(0, polyX1, 0.1) + 3, label="gamma", linestyle='-', c='black')
    ax.plot(np.arange(0, polyX1, 0.1), 2 * np.arange(0, polyX1, 0.1) + 3, label="lognormal", linestyle='-.', c='black')
    ax.legend()
    ax.set_ylim(y_lim, 0)
    ax.set_xlim(-0.2, polyX1)
    plt.xlabel("Skewness²")
    plt.title(title + ": Cullen and Frey map")
    plt.ylabel("Kurtosis")
    plt.savefig('CullenFrey/' + title + ' cullenfrey.png')
    plt.show()
