
from numericalmethods.ode import euler_explicit, euler_implicit, rk4


def test_euler_explicit():

    ans = [1.0, 1.1, 1.22, 1.3620, 1.5282, 1.7210, 1.9431, 2.1974, 2.4872, 2.8159]

    assert [round(sol, 4) for sol in euler_explicit(f=lambda x, y: x+y, y0=1, t0=0, t=1, h=0.1)][:-1] == ans

# TODO: check why this fails (wrong numerical derivative)
# def test_euler_implicit_without_dependance_on_y():

#     ans = [0, 1, 4, 9, 17, 28]

#     assert list(euler_implicit(f=lambda y, t: (1 + t**3)**(1/2), y0=0, t0=0, t=5, h=1)) == ans


def test_runge_kutta4():

    ans = [1.0, 1.11034167, 1.24280514, 1.39971699, 1.58364848, 1.79744128,
           2.04423592, 2.32750325, 2.65107913, 3.01920283, 3.43655949]

    assert [round(i, 4) for i in rk4(lambda y, t : y + t, y0=1, t0=0, t=1, h=0.1)] == [round(i, 4) for i in ans]
