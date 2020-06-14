import process
from process import Spectrum

anatase_theory = Spectrum('anatase_python_test.csv')
anatase_observed = Spectrum('anatase_observed.csv')

# anatase_theory.x_shift(2.61)
# anatase_theory.scale(309.95)
# print(process.residual(anatase_theory, anatase_observed, 0, 9))
# print(process.residual(process.optimise(anatase_theory, anatase_observed, 2.6, 0, 9), anatase_observed))

optimised = process.optimise(anatase_theory, anatase_observed, 0, 9)
print(optimised.data())
