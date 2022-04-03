
import math 

import numpy as np



def forward(order:int, f:'function', x:float, h:float) -> float: 
    raise NotImplementedError()


def backward(order:int, f:'function', x:float, h:float) -> float: 
    raise NotImplementedError()