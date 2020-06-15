import process
from process import Spectrum

theory = Spectrum('anatase_python_test.csv')
observed = Spectrum('anatase_observed.csv')

# Change desired range on the right
optimised = process.optimise(theory, observed, 0, 9)
print(process.residual(optimised, observed, 0 , 9))
print(optimised.get_scale())
print(optimised.get_shift())
