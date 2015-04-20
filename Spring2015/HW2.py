# Problem Set#2
# Kartik Shridhar
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt


def lin_basis(t):
    return [1, t]


def cub_basis(t):
    return [1, t, t**2, t**3]


def spl(t, a):
    return pow(max(0, t+a), 3)


def cub_spl_basis(t):
    return[1, t, t**2, t**3, spl(t, 2), spl(t, 0), spl(t, -2)]


def mse(omega, basisfn, data):
    return np.mean([(sum(omega * basisfn(a[0])) - a[1])**2 for a in data])
    
    
def k_fold (data, k):
    items = list(data)
    shuffle(items)
    slices = [items[i::k] for i in range(k)]

    for i in xrange(k):
         validation = slices[i]
         training = [item
                     for s in slices if s is not validation
                     for item in s]
         yield training, validation






###############################################################################
#
# This is the main() entrypoint of every top-level Python program.
#
# if __name__ == '__main__':
#

sample1_train = "/Users/kartiks/Documents/github/ml/Spring2015/HW1_sample_data/hw1_sample1_train.txt"
sample1_test = "/Users/kartiks/Documents/github/ml/Spring2015/HW1_sample_data/hw1_sample1_test.txt"
sample2_train = "/Users/kartiks/Documents/github/ml/Spring2015/HW1_sample_data/hw1_sample2_train.txt"
sample2_test = "/Users/kartiks/Documents/github/ml/Spring2015/HW1_sample_data/hw1_sample2_test.txt"

data = np.loadtxt(sample1_train)

errors = []
for training, validation in k_fold(data, 5):
    X = [a[0] for a in training]
    Y = [a[1] for a in training]
    w1 = np.linalg.lstsq( [cub_spl_basis(x) for x in X], Y)[0]
    error = mse(w1, cub_spl_basis, training)
    errors.append(error) 
    print "MSE = %s" % error
    
mean_error = np.mean(errors)
print "Mean of the errors = %s" % mean_error

    
 
# plt.plot(t, [sum(w1*lin_basis(a)) for a in t])
 
# X = [a[0] for a in validation]
# Y = [a[1] for a in validation]