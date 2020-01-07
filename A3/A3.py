from typing import Any, Union

import numpy as np
from numpy.core.multiarray import ndarray
from scipy.special import gamma as Y
from scipy.stats import beta
from scipy.special import comb as C
import matplotlib.pyplot as plt

# 1 for heads 0 for tails
N = 160
data = np.random.choice([0, 1], size=N, p= [0.3, 0.7])

posterior = {'a':2 , 'b':3}



def run_sequential(n):
    for i in range(n):
        if data[i] == 0:
            posterior['b'] = posterior['b'] + 1
        else:
            posterior['a'] = posterior['a'] + 1
        # plot
        a = posterior['a']
        b = posterior['b']
        x = np.linspace(0, 1, 1000)
        #y = (Y(a + b) / Y(a) * Y(b)) * (x ** (a - 1)) * ((1 - x) ** (b - 1))
        y = beta.pdf(x,a,b)
        plt.plot(x, y)
        plt.title('Iteration {}'.format(i + 1))
        plt.xlabel('x')
        plt.show(block=False)
        plt.pause(0.00004)
        plt.clf()
        # plt.figure(figsize=(15, 15))
        # plt.plot(x, y)
        # plt.title("Gamma Distribution params a:{} b:{}.Samples:{}".format(a, b, i + 1))
        # plt.savefig("distribution_for_samples_{}".format(i + 1) + str(".png"))
        # plt.show()
        # print(i)
        # plt.close()


run_sequential(N)


def run_once(n):
    m = np.sum(data)
    l = n - m  # type: int
    x = np.linspace(0, 1, 1000)
    a = posterior['a']
    b = posterior['b']
    #y = (Y(a + b) / Y(a) * Y(b)) * (x ** (a - 1)) * ((1 - x) ** (b - 1))
    y = beta.pdf(x,a,b)
    plt.figure(figsize=(15, 15))
    plt.plot(x, y)
    plt.title("Distribution params")
    plt.savefig("LISA_distribution_for_samples_" + str(".png"))
    plt.close()


run_once(N)


def Q3(n):
    #m = np.sum(data)
    #l = n - m  # type: int
    x = np.linspace(0, 1, 1000)
    a = 80
    b = 80
    #y = (Y(a + b) / Y(a) * Y(b)) * (x ** (a - 1)) * ((1 - x) ** (b - 1))
    y = beta.pdf(x,a,b)
    plt.figure(figsize=(15, 15))
    plt.plot(x, y)
    plt.title("Distribution params")
    plt.savefig("Q3_Posterior_distribution_for_samples_" + str(".png"))
    plt.close()

Q3(N)