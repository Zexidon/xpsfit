import numpy as np
np.set_printoptions(suppress=True)


# Shifts the x-axis of the data by the defined amount
def x_shift(data, shift):
    data[:, 0] += shift
    return data


# Scales the intensity of the data by the defined amount
def scale(data, scaling):
    data[:, 1:] *= scaling
    return data


# Reduces the amount of data in the expected data to facilitate fitting to the observed data
# To Do: account for missing x-axis if shift is large
def match(expected, observed, lower, upper):
    reduced = np.zeros((np.size(observed, 0), np.size(expected, 1)))
    i = 0
    for x in observed[:, 0]:
        row = np.where((x + 0.0001 >= expected[:, 0]) & (expected[:, 0] >= x - 0.0001))
        reduced[i] = expected[row, :]
        i += 1
    return reduced


# Calculates the Rf value of the expected vs observed data
def residual(expected, observed, lower, upper):
    observed = range_select(observed, lower, upper)
    expected = match(expected, observed, lower, upper)
    numerator = np.sum(np.abs(np.abs(observed[:, 1]) - np.abs(expected[:, 1])))
    denominator = np.sum(np.abs(observed[:, 1]))
    return (numerator / denominator) * 100


# Trims the observed data to the desired range for fitting
def range_select(data, lower, upper):
    search = data[:, 0]
    top = np.where((upper + 0.0001 >= search) & (search >= upper - 0.0001))
    bottom = np.where((lower + 0.0001 >= search) & (search >= lower - 0.0001))
    ranged = data[top[0][0]:bottom[0][0]+1, :]
    return ranged


anatase_theory = np.genfromtxt('anatase_python_test.csv', delimiter=',')
anatase_theory = anatase_theory[1:]

anatase_observed = np.genfromtxt('anatase_observed.csv', delimiter=',')
anatase_observed = anatase_observed[1:]

corrected = x_shift(scale(anatase_theory, 309.95), 2.61)
print(residual(corrected, anatase_observed, 0, 9))

