
import numpy as np

from numericalmethods.roots import newton
from numericalmethods.integrate import composite_simpson, composite_trapezoid



def test_newton_f_and_f_dev():
    ans = 1
    f = lambda x: x**2 -1
    f_dev = lambda x: 2*x

    assert round(newton(err=1e-10, f=f, f_dev=f_dev, x0=5), 10) == ans

def test_newton_f_dev_and_integrator_composite_trapezoid():
    ans = 1.6448536269886083
    f_dev = lambda x: 1 / np.sqrt(2*np.pi) * np.exp(-x**2 / 2)

    assert newton(err=1e-5, f_dev=f_dev, integrator=composite_trapezoid, c=-0.45) == ans

def test_newton_f_dev_and_integrator_composite_simpson():
    ans = 1.6448536269514884
    f_dev = lambda x: 1 / np.sqrt(2*np.pi) * np.exp(-x**2 / 2)

    assert newton(err=1e-5, f_dev=f_dev, integrator=composite_simpson, c=-0.45) == ans