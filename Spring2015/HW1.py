# Homework #1 ML 302
from import numpy *
import matplotlib.pyplot as plt


def max_cubed(x):
    result = x*x*x
    if (result < 0):
        result = 0
    return result


def CubicSpline(w3, t):
    result = []
    for item in numpy.nditer(t):
        value = w3[0] + w3[1] * item + w3[2] * item * item + \
                w3[3] * item * item * item + \
                w3[4] * max_cubed(item+2) + w3[5] * max_cubed(item) + \
                w3[6] * max_cubed(item-2)
        result.append(value)

    return result

# training set "/Users/kartiks/Documents/github/ml/Spring2015/HW1_sample_data/hw1_sample1_train.txt"


def LinearRegressoin(input_file):
    data = numpy.loadtxt(input_file)
    X = [a[0] for a in data]
    Y = [a[1] for a in data]
    # Matrix
    A = [[1, a] for a in X]
    weights = numpy.linalg.lstsq(A, Y)[0]
    return weights
   
   
   def MeanSquaredError(w):
    testdata = numpy.loadtxt("/Users/kartiks/Documents/github/ml/Spring2015/HW1_sample_data/hw1_sample1_test.txt")
    X = [a[0] for a in testdata]
    Y = [a[1] for a in testdata]
    Y1 = [w[0]+w[1]*a for a in X]
    

###############################################################################
#
# This is the main() entrypoint of every top-level Python program.
#
# if __name__ == '__main__':
#
#  data = numpy.loadtxt("/Users/kartiks/Documents/github/ml/Spring2015/HW1_sample_data/hw1_sample1_train.txt")
#  X = [a[0] for a in data]
#  Y = [a[1] for a in data]
#
#  #Spline Cube
#  X7 = [[1,a,a*a,a*a*a,max_cubed(a+2),max_cubed(a),max_cubed(a-2)] for a in X]
#  w3,_,_,_ = numpy.linalg.lstsq(X7,Y)
#  t = numpy.arange(-4,4,0.1)
#
# spline_values = compute_Y(w3,t)
#
# dummy=1

###############################################################################
###############################################################################

