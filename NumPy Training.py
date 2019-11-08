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
print("_______________DZ__5____________________________")


Z = np.array([1,2,3,4,5])
nz = 3
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
Z0[::nz+1] = Z
print(Z0)
print("_______________________________________________")
vec = np.arange(25).reshape(5,5)
vec[[0,1]] = vec[[1,0]]
print(vec)
print("_______________________________________________")
faces = np.random.randint(0,100,(10,3))
F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print(G)
print("_______________________________________________")
C = np.bincount([1,1,2,3,4,4,6])
A = np.repeat(np.arange(len(C)), C)
print(A)
print("_______________________________________________")
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
print(moving_average(np.arange(20), 3))
print("_______________________________________________")

from numpy.lib import stride_tricks
def rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3)
print(Z)

print("_______________________________________________")
Z = np.random.randint(0,2,100)
np.logical_not(Z, out=Z)
print(Z)

print("_______________________________________________")

def distance(P0, P1, p):
    T = P1 - P0
    L = (T**2).sum(axis=1)
    U = -((P0[:,0] - p[...,0]) * T[:,0] + (P0[:,1] - p[...,1]) * T[:,1]) / L
    U = U.reshape(len(U),1)
    D = P0 + U * T - p
    return np.sqrt((D**2).sum(axis=1))
P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p  = np.random.uniform(-10,10,( 1,2))
print(distance(P0, P1, p))

print("_______________________________________________")
Z = np.random.randint(0,10, (10,10))
shape = (5,5)
fill  = 0
position = (1,1)

R = np.ones(shape, dtype=Z.dtype)*fill
P  = np.array(list(position)).astype(int)
Rs = np.array(list(R.shape)).astype(int)
Zs = np.array(list(Z.shape)).astype(int)

R_start = np.zeros((len(shape),)).astype(int)
R_stop  = np.array(list(shape)).astype(int)
Z_start = (P - Rs//2)
Z_stop  = (P + Rs//2)+Rs%2

R_start = (R_start - np.minimum(Z_start, 0)).tolist()
Z_start = (np.maximum(Z_start, 0)).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
Z_stop = (np.minimum(Z_stop,Zs)).tolist()

r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
R[r] = Z[z]
print(Z)
print(R)


print("_______________________________________________")


Z = np.random.uniform(0,1,(10,10))
rank = np.linalg.matrix_rank(Z)

print("_______________________________________________")
Z = np.random.randint(0,10,50)
print(np.bincount(Z).argmax())

print("_______________________________________________")

Z = np.random.randint(0,5,(10,10))
n = 3
i = 1 + (Z.shape[0] - n)
j = 1 + (Z.shape[1] - n)
C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
print(C)


print("_______________________________________________")




p, n = 10, 20
M = np.ones((p,n,n))
V = np.ones((p,n,1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
print(S)

print("_______________________________________________")


Z = np.ones((16,16))
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                       np.arange(0, Z.shape[1], k), axis=1)


print("_______________________________________________")

Z = np.arange(10000)
np.random.shuffle(Z)
n = 5

print (Z[np.argpartition(-Z,n)[:n]])

print("_______________________________________________")


def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = map(len, arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

print(cartesian(([1, 2, 3], [4, 5], [6, 7])))


print("_______________________________________________")



A = np.random.randint(0,5,(8,3))
B = np.random.randint(0,5,(2,2))

C = (A[..., np.newaxis, np.newaxis] == B)
rows = (C.sum(axis=(1,2,3)) >= B.shape[1]).nonzero()[0]
print(rows)

print("_______________________________________________")

Z = np.random.randint(0,5,(10,3))
E = np.logical_and.reduce(Z[:,1:] == Z[:,:-1], axis=1)
U = Z[~E]
print(Z)
print(U)
print("_______________________________________________")

I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
print(np.unpackbits(I[:, np.newaxis], axis=1))


print("_______________________________________________")

Z = np.random.randint(0, 2, (6,3))
T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
_, idx = np.unique(T, return_index=True)
uZ = Z[idx]
print(uZ)


print("_______________________________________________")

#np.einsum('i->', A)       # np.sum(A)
#np.einsum('i,i->i', A, B) # A * B
#np.einsum('i,i', A, B)    # np.inner(A, B)
#np.einsum('i,j', A, B)    # np.outer(A, B)













