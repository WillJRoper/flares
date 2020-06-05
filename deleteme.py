import numpy as np

def power(x, pow):
    """ A function to calculate powers of the number x.

    :param x: a number [int/float]
    :param pow: The power [int/float]
    :return:
    """
    return x ** pow

# Initialise array to store results
arr = np.zeros(100)

# Loop over numbers
for ind, num in enumerate(range(0, 100)):
    arr[ind] = power(num, 3)  # store result in the array

print(arr)










