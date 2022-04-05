
from charliepy.ode import euler_explicit


def test_euler_explicit():
    
    ans = [1.0, 1.1, 1.22, 1.3620, 1.5282, 1.7210, 1.9431, 2.1974, 2.4872, 2.8159]
    f = lambda x, y: x+y
    
    assert [round(sol, 4) for sol in euler_explicit(f=f, y0=1, t0=0, t=1, h=0.1)] == ans