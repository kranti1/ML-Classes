# Problem Set#2
# Kartik Shridhar
import numpy as np
import math
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


def nat_spl(t, r, i):
    return ((spl(t, r[i]) - spl(t, r[i+1]))/(r[i+1]-r[i]))

def sigmoid (t):
    return (1/(1+ math.e**-t))
    
        


def natural_cub_spl_basis(t, r):
    if len(r) <= 2:
        return [1, t]
    else:
        return [1, t] + [nat_spl(t, r, i) - nat_spl(t, r, i+1) for i in range(len(r) - 2)]


def mse(omega, basisfn, data):
    return np.mean([(sum(omega * basisfn(a[0])) - a[1])**2 for a in data])


def mse_nat_cub_spline(omega, basisfn, knots, data):
    return np.mean([(sum(omega * basisfn(a[0], knots)) - a[1])**2 for a in data])


def BatchGradDesc (basisfn, alpha, data, numIter, numCount, errorlist):

    X = [a[0] for a in data] 
    X1  = [basisfn(a[0]) for a in data]
    omega = np.ones(len(X1[0]))
    Y = [a[1] for a in data]
    m = len(Y)
    count = 0
    for iter in range(0,numIter):
        count = count + 1   
        loss = np.dot(X1,omega) -Y
        omega = omega - alpha * 1/m * (np.dot (loss,X1))  
        if count == numCount:
            errorlist.append( mse (omega,basisfn,data))
            count = 0
 #           print "error list = %s" % errorlist
    return omega
    
    
def k_fold(data, k):
    items = list(data)
    shuffle(items)
    slices = [items[i::k] for i in range(k)]

    for i in xrange(k):
        validation = slices[i]
        training = [item
                    for s in slices if s is not validation
                    for item in s]
        yield training, validation


def RunBatchGradDesc(basisfn, alpha, data, numIter, numCount):
    errorlist = []
    w = BatchGradDesc (basisfn, alpha, data, numIter, numCount, errorlist)
    error = mse (w,basisfn,data)
    return errorlist

def finitelist(l):
    x = l[np.isfinite(l)]
    return x

###############################################################################
#
# This is the main() entrypoint of every top-level Python program.
#
# if __name__ == '__main__':
#

#sample1_train = "/Users/kartiks/Documents/github/ml/Spring2015/HW1_sample_data/hw1_sample1_train.txt"
#sample1_test = "/Users/kartiks/Documents/github/ml/Spring2015/HW1_sample_data/hw1_sample1_test.txt"
#sample2_train = "/Users/kartiks/Documents/github/ml/Spring2015/HW1_sample_data/hw1_sample2_train.txt"
#sample2_test = "/Users/kartiks/Documents/github/ml/Spring2015/HW1_sample_data/hw1_sample2_test.txt"

#       loss = np.dot(X1,omega)-Y 
#       omega = omega - alpha * 1/m * (np.dot (loss,X1))  


sample1_train = "c:/docs/github/ml/Spring2015/HW1_sample_data/hw1_sample1_train.txt"
sample1_test = "c:/docs/github/ml/Spring2015/HW1_sample_data/hw1_sample1_test.txt"
sample2_train = "c:/docs/github/ml/Spring2015/HW1_sample_data/hw1_sample2_train.txt"
sample2_test = "c:/docs/github/ml/Spring2015/HW1_sample_data/hw1_sample2_test.txt"
Lambda = [.001,.01, .1, 1, 10]

errormatrix = np.zeros((5,20))

data = np.loadtxt(sample1_train)

for i in range(0,5):
    errormatrix[i] = RunBatchGradDesc(cub_spl_basis, Lambda[i], data, 1000, 50)




plt.plot( errormatrix[0], 'ro', errormatrix[1],'r--' , errormatrix[2],'g^', \
          errormatrix[3],'bs' , errormatrix[4],'b--')
plt.show()
    
