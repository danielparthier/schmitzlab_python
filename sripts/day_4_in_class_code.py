import numpy as np
import matplotlib.pyplot as plt

simple_array = np.array([1, 2, 3])

array_1d = np.zeros(10)
array_2d = np.zeros((10, 10))
array_3d = np.zeros((10, 10, 10))


# linspace and logspace in 1D and nD
linear_space = np.linspace(start=1, stop=10, num=20)

# where do you see the difference?
log_space = np.logspace(start=1, stop=10, num=20)

geom_space = np.geomspace(start=1, stop=10, num=20)

plt.plot(linear_space, label='linear')
plt.plot(geom_space, label='geom')
plt.legend()
plt.show()


plt.plot(linear_space, label='linear')
plt.plot(log_space, label='log')
plt.legend()
plt.show()

# multi-dimensional linspace
np.linspace(start=np.array([2.2,2]), stop=3, num=10, axis=0)

# Generate random numbers
np.random.normal(5, 2, 10)