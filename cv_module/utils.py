from cv_module.cv_settings import *
import numpy as np
def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def linear_to_exponent(value):
    EXPONENT = 1.25
    LINEAR_MAX_CONVERSION = 4096
    if 0 < value < SQUARE_SIZE:
        if value > SQUARE_SIZE/2:
            value = translate(value,  SQUARE_SIZE/2,  SQUARE_SIZE, 1, LINEAR_MAX_CONVERSION)
            return int(np.float_power(value, EXPONENT))
        else:
            value = translate(value, 0,  SQUARE_SIZE/2, LINEAR_MAX_CONVERSION, 1)

            return int(np.float_power(value, EXPONENT)) * -1
    elif 0 < value:
        return 32700
    else:
        return -32700


def linear_to_limit(value):
    LINEAR_MAX_CONVERSION = 32767
    if 0 < value < SQUARE_SIZE:
        if value > SQUARE_SIZE/2:
            value = translate(value,  SQUARE_SIZE/2,  SQUARE_SIZE, 1, LINEAR_MAX_CONVERSION)
            return int(value)
        else:
            value = translate(value, 0,  SQUARE_SIZE/2, LINEAR_MAX_CONVERSION, 1)

            return int(value) * -1
    elif 0 < value:
        return 32700
    else:
        return -32700
