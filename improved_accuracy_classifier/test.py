# from grader import __main__
#
# __main__.run()

import torch
from torch import nn
# # With square kernels and equal stride
# m = nn.ConvTranspose2d(16, 33, 3, stride=2)
# input = torch.randn(20, 16, 50, 100)
# output = m(input)
# # exact output size can be also specified as an argument
# input = torch.randn(1, 16, 12, 12)
# downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
# upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
# h = downsample(input)
# print(h.size())
# output = upsample(h, output_size=input.size())
# print(output.size())
# aaa = 1


from math import *


def secret_function():
    return "Secret key is 1234"


def function_creator():
    # expression to be evaluated
    expr = input("Enter the function(in terms of x):")

    # variable used in expression
    x = int(input("Enter the value of x:"))

    # passing variable x in safe dictionary
    safe_dict['x'] = x

    # evaluating expression
    y = eval(expr, {"__builtins__": None}, safe_dict)

    # printing evaluated result
    print("y = {}".format(y))


if __name__ == "__main__":
    # list of safe methods
    safe_list = ['acos', 'asin', 'atan', 'atan2', 'ceil', 'cos',
                 'cosh', 'degrees', 'e', 'exp', 'fabs', 'floor',
                 'fmod', 'frexp', 'hypot', 'ldexp', 'log', 'log10',
                 'modf', 'pi', 'pow', 'radians', 'sin', 'sinh', 'sqrt',
                 'tan', 'tanh']

    # creating a dictionary of safe methods
    safe_dict = dict([(k, locals().get(k, None)) for k in safe_list])

    function_creator()