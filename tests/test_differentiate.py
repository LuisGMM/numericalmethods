
from numericalmethods.differentiate import forward, backward, central


# def test_forward_order_1_to_10():

#     ans = [10, 90,  5_040, 30_240, 151_200, 604_800, 1_814_400, 6_628_800, 6_628_800]
#     f = lambda x: x**10

#     assert [round(forward(order=n, f=f, x=1, h=1e-2, exact=True), 5) for n in range(1, 10)] == ans