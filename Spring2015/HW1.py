# Homework #1 ML 302
# Kartik Shridhar
from numpy import *
import matplotlib.pyplot as plt


def max_cubed(x):
    result = x*x*x
    if (result < 0):
        result = 0
    return result


def CubicSpline(w3, t):
    result = []
    for item in nditer(t):
        value = w3[0] + w3[1] * item + w3[2] * item * item + \
                w3[3] * item * item * item + \
                w3[4] * max_cubed(item+2) + w3[5] * max_cubed(item) + \
                w3[6] * max_cubed(item-2)
        result.append(value)

    return result


def LinearRegression(input_file):
    data = loadtxt(input_file)
    X = [a[0] for a in data]
    Y = [a[1] for a in data]
    # Matrix
    A = [[1, a] for a in X]
    weights = linalg.lstsq(A, Y)[0]
    return weights


def LinearMSE(w, test_file):
    testdata = loadtxt(test_file)
    X = [a[0] for a in testdata]
    Y = array([a[1] for a in testdata])
    Y1 = array([w[0]+w[1]*a for a in X])
    mse = mean(square(Y1-Y))
    return mse


def CubicRegression(input_file):
    data = loadtxt(input_file)
    X = [a[0] for a in data]
    Y = [a[1] for a in data]
    # Matrix
    A =  [[1,a, a*a, a*a*a] for a in X]
    weights = linalg.lstsq(A, Y)[0]
    return weights


def CubicMSE(w,test_file):
    testdata = loadtxt(test_file)
    X = [a[0] for a in testdata]
    Y = array([a[1] for a in testdata])
    Y1 = array([w[0]+w[1]*t+w[2]*t*t+w[3]*t*t*t for t in X])
    mse = mean(square(Y1-Y))
    return mse


def CubicSplineRegression(input_file):
    data = loadtxt(input_file)
    X = [a[0] for a in data]
    Y = [a[1] for a in data]
    # Matrix
    A =  [[1, a, a*a, a*a*a, max_cubed(a+2), max_cubed(a), max_cubed(a-2)] for a in X]
    weights = linalg.lstsq(A, Y)[0]
    return weights


def CubicSplineMSE(w,test_file):
    testdata = loadtxt(test_file)
    X = [a[0] for a in testdata]
    Y = array([a[1] for a in testdata])
    Y1 = array(CubicSpline(w,array(X)))
    mse = mean(square(Y1-Y))
    return mse
    
    
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

weights = LinearRegression(sample1_train)
mse = LinearMSE(weights, sample1_test)
print "MSE for sample1 using Linear regression = %s" % mse

weights = LinearRegression(sample2_train)
mse = LinearMSE(weights,sample2_test)
print "MSE for sample2 using Linear regression = %s" % mse

weights = CubicRegression(sample1_train)
mse = CubicMSE(weights, sample1_test)
print "MSE for sample1 using cubic regression = %s" % mse

weights = CubicRegression(sample2_train)
mse = CubicMSE(weights,sample2_test)
print "MSE for sample2 using cubic regression = %s" % mse


weights = CubicSplineRegression(sample1_train)
mse = CubicSplineMSE(weights, sample1_test)
print "MSE for sample1 using cubic spline regression = %s" % mse

weights = CubicSplineRegression(sample2_train)
mse = CubicSplineMSE(weights, sample2_test)
print "MSE for sample2 using cubic spline regression = %s" % mse

###############################################################################
###############################################################################

