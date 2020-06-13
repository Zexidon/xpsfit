import numpy as np
from multipledispatch import dispatch
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
def match(expected, observed):
    reduced = np.zeros((np.size(observed, 0), np.size(expected, 1)))
    i = 0
    for x in observed[:, 0]:
        row = np.where((x + 0.0001 >= expected[:, 0]) & (expected[:, 0] >= x - 0.0001))
        reduced[i] = expected[row, :]
        i += 1
    return reduced


# Calculates the Rf value of the expected vs observed data
@dispatch(np.ndarray, np.ndarray)
def residual(expected, observed):
    expected = match(expected, observed)
    numerator = np.sum(np.abs(np.abs(observed[:, 1]) - np.abs(expected[:, 1])))
    denominator = np.sum(np.abs(observed[:, 1]))
    return (numerator / denominator) * 100


# Calculates the Rf value of the expected vs observed data
# Only calculates within the desired range
@dispatch(np.ndarray, np.ndarray, int, int)
def residual(expected, observed, lower, upper):
    observed = range_select(observed, lower, upper)
    return residual(expected, observed)


# Calculates the Rf value of the expected vs observed data
# Only calculates within the desired range
@dispatch(np.ndarray, np.ndarray, float, float)
def residual(expected, observed, lower, upper):
    if len(str(lower)) > 3 or len(str(upper)) > 3:
        return print('Please limit range to one decimal point')
    observed = range_select(observed, lower, upper)
    return residual(expected, observed)


# Calculates the Rf value of the expected vs observed data
# Only calculates within the desired range
@dispatch(np.ndarray, np.ndarray, int, float)
def residual(expected, observed, lower, upper):
    if len(str(upper)) > 3:
        return print('Please limit range to one decimal point')
    observed = range_select(observed, lower, upper)
    return residual(expected, observed)


# Calculates the Rf value of the expected vs observed data
# Only calculates within the desired range
@dispatch(np.ndarray, np.ndarray, float, int)
def residual(expected, observed, lower, upper):
    if len(str(lower)) > 3:
        return print('Please limit range to one decimal point')
    observed = range_select(observed, lower, upper)
    return residual(expected, observed)


# Trims the observed data to the desired range for fitting
def range_select(data, lower, upper):
    search = data[:, 0]
    top = np.where((upper + 0.0001 >= search) & (search >= upper - 0.0001))
    bottom = np.where((lower + 0.0001 >= search) & (search >= lower - 0.0001))
    ranged = data[top[0][0]:bottom[0][0]+1, :]
    return ranged
