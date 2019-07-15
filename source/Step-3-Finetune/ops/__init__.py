'''
Pytorch Custom Ops
'''
from .average_pool2d import AveragePool2d
from .bitwise import lshift, rshift
from .median_pool2d import MedianPool2d
from .power import op_pow_2, PowerOfTwo, PowerOfTwoFunction
from .roundoff import Ceil, CeilFunction, op_ceil
from .roundoff import Floor, FloorFunction, op_floor
from .roundoff import Round, RoundFunction, op_round
from .variance_pool2d import VariancePool2d
