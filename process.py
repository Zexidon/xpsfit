import numpy as np

np.set_printoptions(suppress=True)


# A spectrum class that stores the original spectroscopic information
# along with the modified spectrum, to be able to return the scale/energy shift
# If passed a csv file, also stores the column titles
class Spectrum:
    # Constructor takes a file path text input, or an array input
    def __init__(self, csvfile):
        if type(csvfile) == str:
            array = np.genfromtxt(csvfile, delimiter=',')
            titles = np.genfromtxt(csvfile, delimiter=',', dtype=str)
            self.original = np.copy(array[1:])
            self.spectrum = np.copy(array[1:])
            self.titles = titles[0]
        elif type(csvfile) == np.ndarray:
            self.original = csvfile
            self.spectrum = csvfile
            self.titles = None

    # Returns the current spectrum (with any modifications applied)
    def data(self):
        return self.spectrum

    # Returns the original unmodified spectrum
    def original(self):
        return self.original

    # Returns the scaling factor of the current spectrum
    def get_scale(self):
        scale_factor = self.spectrum[:, 1] / self.original[:, 1]
        return np.average(scale_factor)

    # Returns the energy shift of the current spectrum
    def get_shift(self):
        shift = self.spectrum[:, 0] - self.original[:, 0]
        return np.average(shift)

    # Shifts the x-axis of this spectrum by the defined amount
    def x_shift(self, shift):
        self.spectrum[:, 0] += shift
        return self.spectrum

    # Scales the intensity of this spectrum by the defined amount
    def scale(self, scaling):
        self.spectrum[:, 1:] = self.original[:, 1:] * scaling
        return self.spectrum


# Shifts the x-axis of the data by the defined amount
def x_shift(data, shift):
    data.x_shift(shift)


# Scales the intensity of the data by the defined amount
def scale(data, scaling):
    data.scale(scaling)


# Reduces the amount of data in the expected spectrum to facilitate fitting to the observed data
# To Do: account for missing x-axis if shift is large
def __match(expected, observed):
    # Creates an empty array with a size of the final array to be produced
    row_size = np.size(observed.data(), 0)
    column_size = np.size(expected.data(), 1)
    reduced = np.zeros((row_size, column_size), dtype=float)
    i = 0
    # Searches the expected spectrum for each x-axis point in observed spectrum
    # Takes the row with that x-axis point and inserts it into the empty array
    for x in observed.data()[:, 0]:
        row = np.where((x + 0.00001 >= expected.data()[:, 0]) & (expected.data()[:, 0] >= x - 0.00001))
        reduced[i] = expected.data()[row, :]
        i += 1
    return Spectrum(reduced)


# Trims the observed spectrum to the desired range for fitting
def __range_select(spectrum, lower, upper):
    # Raises exceptions if the range input is invalid
    # Range has to be two numbers, up to 1 d.p.
    if isinstance(lower, (int, float)) == False or isinstance(upper, (int, float)) == False:
        raise Exception('Please input two numbers (up to one decimal point)')
    elif upper <= lower:
        raise Exception('Upper bound must be smaller than lower bound')
    elif len(str(lower)) > 3 or len(str(upper)) > 3:
        raise Exception('Please limit range values to one decimal point')
    search = spectrum.data()[:, 0]
    top = np.where((upper + 0.0001 >= search) & (search >= upper - 0.0001))
    bottom = np.where((lower + 0.0001 >= search) & (search >= lower - 0.0001))
    trimmed = spectrum.data()[top[0][0]:bottom[0][0] + 1, :]
    return Spectrum(trimmed)


# Calculates the Rf value of the expected vs observed data
def residual(expected, observed, lower='min', upper='max'):
    # If a range has been defined, trim the spectrum to the selected range
    # Otherwise, use the full range
    if lower != 'min' or upper != 'max':
        try:
            trimmed = __range_select(observed, lower, upper)
        except Exception as e:
            return print(e)
    else:
        trimmed = observed
    reduced = __match(expected, trimmed)
    numerator = np.sum(np.abs(np.abs(trimmed.data()[:, 1]) - np.abs(reduced.data()[:, 1])))
    denominator = np.sum(np.abs(trimmed.data()[:, 1]))
    return (numerator / denominator) * 100


# Private method for optimising the energy shift automatically
def __optimise_x(expected, observed, step_size):
    # Calculates the current Rf and set it as the min for baseline value
    # Then shifts it by one step and recalculates the Rf
    rf = residual(expected, observed)
    min_rf = float(rf)
    expected.x_shift(step_size)
    rf = residual(expected, observed)
    # If the Rf is smaller, continue shifting it until the Rf gets bigger
    # Undo the very last step to get it back to the correct minimum value
    if rf < min_rf:
        while rf < min_rf:
            min_rf = float(rf)
            expected.x_shift(step_size)
            rf = residual(expected, observed)
            if rf > min_rf:
                expected.x_shift(-step_size)
    # If the Rf is bigger, shift it the other way and recalculate the Rf as it gets reduced
    # Once the Rf gets bigger again rather than smaller, undo the very last step
    elif rf > min_rf:
        while rf >= min_rf:
            expected.x_shift(-step_size)
            rf = residual(expected, observed)
            if rf < min_rf:
                min_rf = float(rf)
            elif rf > min_rf:
                expected.x_shift(step_size)
                break
    return expected


# Private method for optimising the intensity scaling automatically
def __optimise_scale(expected, observed, step_size):
    # Calculates the current Rf and set it as the min for baseline value
    # Also retrieves the current scaling of the spectrum
    # Changes the scaling of the spectrum by increasing it by one step
    # Then recalculates the Rf value
    current_scale = expected.get_scale()
    rf = residual(expected, observed)
    min_rf = float(rf)
    current_scale += step_size
    expected.scale(current_scale)
    rf = residual(expected, observed)
    # If the Rf value is smaller, keep increasing the scaling in step sizes
    # Until the Rf gets bigger than minimum, then reverse the very last step
    if rf < min_rf:
        while rf < min_rf:
            min_rf = float(rf)
            current_scale += step_size
            expected.scale(current_scale)
            rf = residual(expected, observed)
            if rf > min_rf:
                current_scale -= step_size
                expected.scale(current_scale)
    # If the Rf value gets bigger, decrease the scaling in step sizes
    # Unless doing so would make it a negative scaling, then undo that and stop
    # Keep decreasing the scaling otherwise until the Rf value gets bigger again
    # Then reverse the very last step
    elif rf > min_rf:
        while rf >= min_rf:
            current_scale -= step_size
            if current_scale <= 0:
                current_scale += step_size
                break
            expected.scale(current_scale)
            rf = residual(expected, observed)
            if rf < min_rf:
                min_rf = float(rf)
            elif rf > min_rf:
                current_scale += step_size
                expected.scale(current_scale)
                break
    return expected


# Automatically optimises the energy shift and scaling of the theoretical data
# To minimise the Rf value when compared to a given set of observed data
def optimise(expected, observed, lower='min', upper='max'):
    if lower != 'min' or upper != 'max':
        try:
            trimmed = __range_select(observed, lower, upper)
        except Exception as e:
            return print(e)
    else:
        trimmed = observed
    i = 1
    # The automatic fitting loops 3 times to ensure good minimisation
    # Different step sizes called each time just to speed up process
    while i <= 3:
        expected = __optimise_x(expected, trimmed, 1)
        expected = __optimise_x(expected, trimmed, 0.1)
        expected = __optimise_x(expected, trimmed, 0.01)
        expected = __optimise_scale(expected, trimmed, 100)
        expected = __optimise_scale(expected, trimmed, 10)
        expected = __optimise_scale(expected, trimmed, 1)
        expected = __optimise_scale(expected, trimmed, 0.1)
        expected = __optimise_scale(expected, trimmed, 0.01)
        i += 1
    return expected
