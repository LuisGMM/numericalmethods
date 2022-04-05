from charliepy.roots import newton

def test_newton():
    ans = 1
    f = lambda x: x**2 -1
    f_dev = lambda x: 2*x
    assert round(newton(err=1e-10, f=f, f_dev=f_dev, x0=5), 10) == ans
