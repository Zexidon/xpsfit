import process
from numpy import genfromtxt as csvdata

anatase_theory = csvdata('anatase_python_test.csv', delimiter=',')
anatase_theory = anatase_theory[1:]

anatase_observed = csvdata('anatase_observed.csv', delimiter=',')
anatase_observed = anatase_observed[1:]

corrected = process.x_shift(process.scale(anatase_theory, 309.95), 2.61)
print(process.residual(corrected, anatase_observed, 0, 9))
