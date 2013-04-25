# !/usr/local/bin/python
'''
Created on Apr 25, 2013

@author: paul
'''
import math, cmath
import numpy as np
import matplotlib.pyplot as plt

class Graph(object):
    def __init__(self, links):
        self.links = links
        self.cleanLinks()
        self.nodes = [elem for tup in links for elem in tup]

    def cleanLinks(self):
        self.links = list( set( self.links ) )
        for (i, j) in self.links:
            if (j, i) in self.links:
                self.links.remove( (j, i) )

    def addLink(self, fromIdx, toIdx):
        if not self.areConnected(fromIdx, toIdx):
            self.links.append( (fromIdx, toIdx) )
            if not fromIdx in self.nodes:
                self.nodes.append( fromIdx)
            if not toIdx in self.nodes:
                self.nodes.append( toIdx )
            
    def areConnected(self, idx1, idx2):
        return len( [x for x in self.links if (idx1 in x and idx2 in x)] ) > 0
    
    def rank(self, node):
        return len( [x for x in self.links if ( node in x )] )
        
    def clusteringCoefficient(self, node, zeroConvention = 'Shanahan'):
        def nonnode((a, b)):
            if a == node:
                return b
            else:
                return a
        neighbors = [nonnode(x) for x in self.links if x[0] == node or x[1] == node]
        d = len(neighbors) * (len(neighbors) - 1) / 2.0
        if len(neighbors) == 1:
            if zeroConvention == 'MATLAB':
                return 0
            else:
                return 1
        else:
            return len( [(i, j) for (i, j) in self.links if i in neighbors and j in neighbors] ) / d
    

def rhythmBand(hz):
    if hz >= 4 and hz <= 8:
        return 'theta'
    elif hz <= 15:
        return 'alpha'
    elif hz <= 30:
        return 'beta'
    elif hz <= 80:
        return 'gamma'

def length(z):
    return math.sqrt( z.real*z.real + z.imag * z.imag )

def demean(time_series):
    avg = time_series.mean()
    assert( (time_series - avg).mean() <= 0.00001 )
    return time_series - avg


def synchrony(thetas):
    result = 0
    for theta in thetas:
        result += cmath.exp( complex( 0, theta ) )
    result /= len( thetas )
    return length( result )

def weightUpdate(dt, Aplus = 1, Aminus = 1, tauPlus = 1, tauMinus = 1):
    if dt >= 0:
        return Aplus * math.exp( -dt / tauPlus )
    elif dt < 0:
        return -Aminus * math.exp( dt / tauMinus )

def kuramotoSim(K, alpha, omega, N, maxT, dt):
    nSteps = int(maxT / dt)
    theta = np.random.rand(nSteps, N) * 2 - 1
    for t in range(1, nSteps):
        for i in range(N):
            sines = np.zeros((N,1))
            for j in range(N):
                sines[j] = K * math.sin(theta[t-1, j] - theta[t-1, i] - alpha)
            theta[t, i] = theta[t-1, i] + dt *(omega + sines.sum() / (N + 1))
    return theta

def entropy(series):
    """ 1/2 ln( (2pi * e)^N |COV_MATRIX(series)| """
    pass

def izhikevichSim(v0, u0, I, a, b, c, d, N, dt):
    v = np.zeros((N, 1))
    u = np.zeros((N, 1))
    v[0] = v0
    u[0] = u0
    
    for i in range(1, N):
        u[i] = u[i-1] + dt * ( a * ( b * v[i-1] - u[i-1] ) )
        v[i] = v[i-1] + dt * ( 0.04*v[i-1]*v[i-1] + 5*v[i-1] + 140 - u[i] + I )
        if v[i] > 30:
            v[i] = c
            u[i] = u[i-1] + d
    return (u, v)

def tests():
    synchTheta = [0.7854, 1.5708, 2.3562]
    print synchrony(synchTheta)
    
    ts = np.random.rand(100, 1)
#     print ts.mean()
#     print demean(ts).mean()
    
    new_ts = (np.random.rand(100, 1) * math.pi)
#     print synchrony(new_ts)
    
    old_ts = np.random.rand(100, 1) * math.pi * 2 - math.pi
#     print synchrony(old_ts)
    
    
#     alpha = 0
#     K = 0.02
#     omega = 0.1
#     sim = kuramotoSim(K, alpha, omega, 10, 1, 0.001)
#     for i in range(sim.shape[0]):
#         print synchrony(sim[i,])
    
    izSim = izhikevichSim(-65, 0, 5, 0.07, 0.25, -65, 8, 1000, 0.1)
    izSim2 = izhikevichSim(-65, 0, 5, 0.1, 0.25, -65, 8, 1000, 0.1)
#     print izSim[1]
#     plt.plot(range(1000), izSim[0],color='red')
#     plt.plot(range(1000), izSim2[0],color='blue')
    print len(izSim[1][izSim[1] > 0])
    
    links = zip( np.random.randint(0, 100, 50).tolist(), np.random.randint(0, 100, 50).tolist() )
    links = [(x, y) for (x, y) in links if x != y]
    links = [(5,6), (6,4), (4,1), (5,4), (1,2), (1,3), (3,2)]
    g = Graph(links)
    
    dt = np.linspace(-5, 5, 1000)
    dw = map( weightUpdate, dt  )
    plt.plot( dt, dw )
    plt.show()

    
tests()
    