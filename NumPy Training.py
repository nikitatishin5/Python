import numpy as np
print(np.__version__)
np.show_config()
vec = np.zeros(10,int)
print(vec)
vec = np.ones(10,int)
print(vec)
vec = np.linspace(2.5,2.5,10)
print(vec)
#python3 -c "import numpy; numpy.info(numpy.add)"
vec = np.zeros(10,int)
vec[4] = 1
print(vec)
vec = np.arange(10,50,1)
print(vec)
vec = np.array([12,58,325,21,6,51,64,84,651,65,498,46,51,651,89,4,84,32,16854])
vec = vec[::-1]
print(vec)
vec = np.arange(9).reshape(3,3)
print(vec)
vec = np.nonzero([1,2,0,0,4,0])
print(vec)
vec = np.random.random((3,3,3))
print(vec)
vec = np.random.random((10,10))
vecmin, vecmax = vec.min(), vec.max()
print(vecmin, vecmax)
vec = np.random.random(30)
m = vec.mean()
print(m)
vec = np.ones((10,10))
vec[1:-1,1:-1] = 0
print(vec)

print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(0.3 == 3 * 0.1)

vec = np.diag(np.arange(1, 5), k=-1)
print(vec)

vec = np.zeros((8,8), dtype=int)
vec[1::2,::2] = 1
vec[::2,1::2] = 1
print(vec)

print(np.unravel_index(100, (6,7,8)))

vec = np.tile(np.array([[0,1],[1,0]]), (4,4))
print(vec)

