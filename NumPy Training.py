import numpy as np
import scipy.spatial

print(np.__version__)
print("_______________________________________________")
np.show_config()
print("_______________________________________________")
vec = np.zeros(10,int)
print(vec)
print("_______________________________________________")
vec = np.ones(10,int)
print(vec)
print("_______________________________________________")
vec = np.linspace(2.5,2.5,10)
print(vec)
print("_______________________________________________")
#python3 -c "import numpy; numpy.info(numpy.add)"
vec = np.zeros(10,int)
print("_______________________________________________")
vec[4] = 1
print(vec)
print("_______________________________________________")
vec = np.arange(10,50,1)
print(vec)
print("_______________________________________________")
vec = np.array([12,58,325,21,6,51,64,84,651,65,498,46,51,651,89,4,84,32,16854])
vec = vec[::-1]
print(vec)
print("_______________________________________________")
vec = np.arange(9).reshape(3,3)
print(vec)
print("_______________________________________________")
vec = np.nonzero([1,2,0,0,4,0])
print(vec)
print("_______________________________________________")
vec = np.random.random((3,3,3))
print(vec)
print("_______________________________________________")
vec = np.random.random((10,10))
vecmin, vecmax = vec.min(), vec.max()
print(vecmin, vecmax)
print("_______________________________________________")
vec = np.random.random(30)
m = vec.mean()
print(m)
print("_______________________________________________")
vec = np.ones((10,10))
vec[1:-1,1:-1] = 0
print(vec)
print("_______________________________________________")

print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(0.3 == 3 * 0.1)
print("_______________________________________________")

vec = np.diag(np.arange(1, 5), k=-1)
print(vec)
print("_______________________________________________")

vec = np.zeros((8,8), dtype=int)
vec[1::2,::2] = 1
vec[::2,1::2] = 1
print(vec)
print("_______________________________________________")

print(np.unravel_index(100, (6,7,8)))
print("_______________________________________________")

vec = np.tile(np.array([[0,1],[1,0]]), (4,4))
print(vec)
print("_______________________________________________")





print("##########____________DZ_4______________######################")





vec = np.dot(np.ones((5,3)), np.ones((3,2)))
print(vec)
print("_______________________________________________")

vec = np.arange(11)
vec[(3 < vec) & (vec <= 8)] *= -1
print(vec)
print("_______________________________________________")


vec = np.zeros((5,5))
vec += np.arange(5)
print(vec)
print("_______________________________________________")


def generate():
    for x in range(10):
        yield x
vec = np.fromiter(generate(),dtype=float,count=-1)
print(vec)
print("_______________________________________________")


vec = np.linspace(0,1,12)[1:-1]
print(vec)
print("_______________________________________________")


vec = np.random.random(10)
vec.sort()
print(vec)
print("_______________________________________________")

vec1 = np.random.randint(0,2,5)
vec2 = np.random.randint(0,2,5)
equal = np.allclose(vec1,vec2)
print(equal)
print("_______________________________________________")

vec = np.zeros(10)
vec.flags.writeable = False
#vec[0] = 1
print("_______________________________________________")

vec = np.random.random((10,2))
X,Y = vec[:,0], vec[:,1]
R = np.hypot(X, Y)
T = np.arctan2(Y,X)
print(R)
print(T)

print("_______________________________________________")


vec = np.random.random(10)
vec[vec.argmax()] = 0
print(vec)

print("_______________________________________________")

vec = np.zeros((10,10), [('x',float),('y',float)])
vec['x'], vec['y'] = np.meshgrid(np.linspace(0,1,10),
                             np.linspace(0,1,10))
print(vec)
print("_______________________________________________")

X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))

print("_______________________________________________")

for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)

print("____________________tyr___________________________")

np.set_printoptions(threshold=1000)
vec = np.zeros((25,25))
print(vec)
print("_______________________________________________")


vec = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(vec-v)).argmin()
print(vec[index])
print("_______________________________________________")


vec = np.zeros(10, [ ('position', [ ('x', float, 1),
                                   ('y', float, 1)]),
                    ('color',    [ ('r', float, 1),
                                   ('g', float, 1),
                                   ('b', float, 1)])])
print(vec)
print("_______________________________________________")




vec = np.random.random((10,2))
D = scipy.spatial.distance.cdist(vec,vec)
print(D)
print("_______________________________________________")



vec = np.arange(10, dtype=np.int32)
print(vec)
vec = vec.astype(np.float32, copy=False)
print(vec)

print("____________________reading___________________________")

vec = np.genfromtxt("text.dat", delimiter=",")
print(vec)
print("_______________________________________________")



Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
    print(index, value)
for index in np.ndindex(Z.shape):
    print(index, Z[index])
print("_______________________________________________")



X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.hypot(X, Y)
sigma, mu = 1.0, 0.0
G = np.exp(-((D - mu) ** 2 / (2.0 * sigma ** 2)))
print(G)
print("_______________________________________________")



n = 10
p = 3
vec = np.zeros((n,n))
np.put(vec, np.random.choice(range(n*n), p, replace=False), 1)
print(vec)
print("_______________________________________________")



X = np.random.rand(5, 10)
print(X)
Y = X - X.mean(axis=1, keepdims=True)
print(Y)
print("_______________________________________________")


vec = np.random.randint(0,10,(3,3))
n = 1  # Нумерация с нуля
print(vec)
print(vec[vec[:,n].argsort()])
print("_______________________________________________")


vec = np.random.randint(0,3,(3,10))
print((~vec.any(axis=0)).any())
print("_______________________________________________")



vec = np.ones(10)
I = np.random.randint(0,len(vec),20)
vec+= np.bincount(I, minlength=len(vec))
print(vec)
print("_______________________________________________")


w,h = 16,16
I = np.random.randint(0, 2, (h,w,3)).astype(np.ubyte)
F = I[...,0] * 256 * 256 + I[...,1] * 256 + I[...,2]
n = len(np.unique(F))
print(np.unique(I))
print("_______________________________________________")



vec = np.random.randint(0,10, (3,4,3,4))
sum = vec.reshape(vec.shape[:-2] + (-1,)).sum(axis=-1)
print(sum)
print("_______________________________________________")


#np.diag(np.dot(A, B))
print("_______________END____________________________")







