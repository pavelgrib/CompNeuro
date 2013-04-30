# !/usr/local/bin/python
'''
Created on Apr 25, 2013

@author: paul
'''
import math, cmath, operator
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mulTup(tup, times):
    return tuple( map( lambda x: x*times, tup))

def addTup(tup1, tup2):
    return tuple(map(operator.add, tup1, tup2)) 

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
        neighbors = self.neighbors(node)
        d = len(neighbors) * (len(neighbors) - 1) / 2.0
        if len(neighbors) == 1:
            if zeroConvention == 'MATLAB':
                return 0
            else:
                return 1
        else:
            return len( [(i, j) for (i, j) in self.links if i in neighbors and j in neighbors] ) / d
    
    def neighbors(self, ofNode):
        def nonnode((a, b)):
            if a == ofNode:
                return b
            else:
                return a
            return [nonnode(x) for x in self.links if 
                    x[0] == ofNode or x[1] == ofNode]
    
    def pathLength(self, fromNode, toNode):
        inf = float('inf')
        dist = dict(zip(self.nodes, [inf]*len(self.nodes)))
        dist[fromNode] = 0
#         Q = self.nodes[:]   # creates a copy of the nodes
#         prev = []

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

def variance(series):
    pd.Series(series).var()

def covariance(S):
    """ assuming that S is a set of time-series with T x N size """
    N = S.shape[1]
    covar = np.zeros(shape=(N, N))
    for i in range(N):
        iSeries = pd.Series(S[i])
        for j in range(N):
            covar[i,j] = iSeries.cov(pd.Series(S[j]))
    return covar

def entropy(S):
    """ 1/2 ln( (2pi * e)^N |COV_MATRIX(S)| """
    covDet = np.linalg.det(covariance(S))
    return math.log(math.pow(2*math.pi*math.e, S.shape[1]) * covDet)

def mutualInformation(S, i):
    """ MI(S, i) = H(S-{X_i}) + H(X_i) - H(S) """
    X = S[:,i]
    return entropy(X) + entropy(S[:,range(len(X)).remove(i)])

def integration(S):
    """ I(S) = (sum H(X_i)) - H(S) """ 
    isum = 0
    for i in range(S.shape[1]):
        isum += entropy(S[:,i])
    return isum - entropy(S)
    
def complexity(S):
    """ C(S) = (sum MI(S, i)) - I(S) """
    csum = 0
    for i in range(S.shape[1]):
        csum += mutualInformation(S, i)
    return csum - integration(S)
    
def izhikevichSim(v0, u0, I, a, b, c, d, N, dt, update='Euler'):
    v = np.zeros((N, 1))
    u = np.zeros((N, 1))
    v[0] = v0
    u[0] = u0
    
    def izUpdate(recov, volts, current):
        du =  a * ( b * volts - recov )
        dv = 0.04*volts*volts + 5*volts + 140 - recov + current
        return du, dv
    
    for i in range(1, N):
        k1u, k1v = izUpdate(u[i-1], v[i-1], I)
        if update.lower()=='euler':
            u[i] = u[i-1] + dt * k1u
            v[i] = v[i-1] + dt * k1v
        elif update.lower()=='rk' or update.lower()=='runge-kutta' or update.lower()=='rungekutta':
            inputs = addTup( (u[i-1], v[i-1]), mulTup( (k1u, k1v), dt/2.0 ) )
            k2u, k2v = izUpdate(inputs[0], inputs[1], I)
            inputs = addTup( (u[i-1], v[i-1]), mulTup( (k2u, k2v), dt/2.0 ) )
            k3u, k3v = izUpdate(inputs[0], inputs[1], I)
            inputs = addTup( (u[i-1], v[i-1]), mulTup( (k3u, k3v), dt ) )
            k4u, k4v = izUpdate(inputs[0], inputs[1], I)
            v[i] = v[i-1] + dt*(k1v + 2*k2v + 2*k3v + k4v) / 6.0
        else:
            print 'update method ' + str(update) + ' not understood'
            
        u[i] = u[i-1] + dt * ( a * ( b * v[i-1] - u[i-1] ) )
        v[i] = v[i-1] + dt * ( 0.04*v[i-1]*v[i-1] + 5*v[i-1] + 140 - u[i] + I )
        if v[i] > 30:
            v[i] = c
            u[i] = u[i-1] + d
    return (u, v)

def eulerUpdate(lastVal, updateFunc, dt):
    return updateFunc(lastVal) * dt

def rkUpdate(lastVal, updateFunc, dt):
    k1 = updateFunc(lastVal)
    k2 = updateFunc(lastVal + k1 * dt / 2)
    k3 = updateFunc(lastVal + k2 * dt / 2)
    k4 = updateFunc(lastVal + k3 * dt    )
    return dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0

def hhSim(v0, m0, n0, h0, N, dt, I, update='Euler', C=1):
    v = np.zeros((N,1), dtype=float)
    m = float(m0)
    n = float(n0)
    h = float(h0)
    v[0] = v0
    
    alpha_m = lambda volts: (2.5 - 0.1*volts) /(math.exp(2.5-0.1*volts) - 1)
    alpha_n = lambda volts: (0.1 - 0.01*volts)/(math.exp(1.0-0.1*volts) - 1)    
    alpha_h = lambda volts: 0.07  * math.exp(-volts/20.0)
    beta_m  = lambda volts: 4.00  * math.exp(-volts/18.0)
    beta_n  = lambda volts: 0.125 * math.exp(-volts/80.0)
    beta_h  = lambda volts: 1.0   / (math.exp(3.0 - 0.1*volts)+1)
    g = np.array([120, 36, 0.3])
    E = np.array([115, -12, 10.6])
    
    def updateM(last_m, volts):
        return alpha_m(volts)*(1-last_m) - beta_m(volts)*last_m
    def updateN(last_n, volts):
        return alpha_n(volts)*(1-last_n) - beta_n(volts)*last_n
    def updateH(last_h, volts):
        return alpha_h(volts)*(1-last_h) - beta_h(volts)*last_h
    def hhUpdate(volts, last_m, last_n, last_h, curr): 
        return curr - sum( g *(volts - E) * 
                           np.array([last_h*math.pow(last_m,3), math.pow(last_n,4), 1]) )
    
#     forPlotting = partial( updateM, 0.1 )
#     plt.plot(np.linspace(-10, 40, 1000), map( forPlotting, np.linspace(-10, 40, 1000) ) )
#     plt.show()

    for i in range(1, N):
#         print v[i-1]
        mUpdater = partial(updateM, volts=v[i-1])
        nUpdater = partial(updateN, volts=v[i-1])
        hUpdater = partial(updateH, volts=v[i-1])
        vUpdater = partial(hhUpdate, last_m=m, last_n=n, last_h=h, curr=I)
        if update.lower() == 'euler':
            m    += eulerUpdate(m, mUpdater, dt)
            n    += eulerUpdate(n, nUpdater, dt)
            h    += eulerUpdate(h, hUpdater, dt)
            v[i] = v[i-1] + eulerUpdate(v[i-1], vUpdater, dt) / C
        elif update.lower()=='rk' or update.lower()=='runge-kutta' or update.lower()=='rungekutta':
            m    += rkUpdate(m, mUpdater, dt)
            n    += rkUpdate(n, nUpdater, dt)
            h    += rkUpdate(h, hUpdater, dt)
            v[i] = v[i-1] + rkUpdate(v[i-1], vUpdater, dt) / C
        else: 
            print 'update method ' + str(update) + ' not understood'
    return v

def LIFSim(v0, N, dt, I, update='Euler', R=1, tau=5, vr=-65, theta=-50, alpha=0.5, spikeV=30):
    v = np.zeros((N, 1), dtype=float)
    v[0] = v0
    t_spike = float('-inf')
    
    def lifUpdate(last_v):
        return vr - last_v + R*I

    for i in range(1, N):
        if update.lower()=='euler':
            temp = v[i-1] + eulerUpdate( v[i-1], lifUpdate, dt ) / tau
        elif update.lower()=='rk' or update.lower()=='runge-kutta' or update.lower()=='rungekutta':
            temp = v[i-1] + rkUpdate(v[i-1], lifUpdate, dt) / tau
        else:
            print 'update method ' + str(update) + ' not understood'
        
        if temp > theta and (i*dt - t_spike) > alpha:
            v[i]   = vr
            v[i-1] = spikeV
            t_spike = i*dt
        else:
            v[i] = temp
    return v

def QIFSim(v0, N, dt, I, update='Euler', R=1, a=0.04, tau=5, vr=-65, vc=-55, theta=-30, alpha=0, spikeV=30):
    v = np.zeros((N, 1), dtype=float)
    v[0] = v0
    t_spike = float('-inf')
    def qifUpdate(last_v):
        return a*(vr - last_v)*(vc - last_v) + R * I
    
    for i in range(1, N):
        if update.lower()=='euler':
            temp = v[i-1] + eulerUpdate( v[i-1], qifUpdate, dt) / tau
        elif update.lower()=='rk' or update.lower()=='runge-kutta' or update.lower()=='rungekutta':
            temp = v[i-1] + rkUpdate(v[i-1], qifUpdate, dt) / tau
        else:
            print 'update method ' + str(update) + ' not understood'
        if temp > theta and (i*dt - t_spike) > alpha:
            v[i  ] = vr
            v[i-1] = spikeV
            t_spike = i*dt
        else:
            v[i] = temp
    return v
     
def numSpikes(v, threshold=0):
    above = v > threshold
    return len( above[above[0:(len(above)-2)] != above[1:(len(above)-1)]] )/2
     
def display(v):
    plt.plot(range(len(v)), v, color='b')
    plt.show()
    
def displayIz((u,v)):
    f, ax = plt.subplots(2, sharex=True)
    ax[0].plot(range(len(v)), v)
    ax[0].set_title('izhikevich neuron firing')
    ax[1].plot(range(len(u)), u)
    plt.show()
    
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
    
#     display( hhSim(-10, 0, 0, 0, 1000, 0.1, 10) )
    lif = LIFSim(-70, 1000, 0.1, 20, update='rk')
    qif = QIFSim(-50, 1000, 0.1, 3, update='rk', spikeV = 20)
    hh  =  hhSim(0, 0, 0, 0, 1000, 0.1, 10, update='rk', C = 3)
    iz_inh = izhikevichSim(v0=-65, u0=0, I=5, a=0.02, b=0.25, c=-65, d=2, N=1000, dt=0.1)
    print numSpikes(qif)
    print numSpikes(lif)
    print numSpikes(hh, 50)
#     display( qif )
    displayIz(iz_inh)
#     alpha = 0
#     K = 0.02
#     omega = 0.1
#     sim = kuramotoSim(K, alpha, omega, 10, 1, 0.001)
#     for i in range(sim.shape[0]):
#         print synchrony(sim[i,])
    
    
#     izSim2 = izhikevichSim(-65, 0, 5, 0.1, 0.25, -65, 8, 1000, 0.1)
#     print izSim[1]
#     plt.plot(range(1000), izSim[0],color='red')
#     plt.plot(range(1000), izSim2[0],color='blue')
     
    
#     links = zip( np.random.randint(0, 100, 50).tolist(), np.random.randint(0, 100, 50).tolist() )
#     links = [(x, y) for (x, y) in links if x != y]
#     links = [(5,6), (6,4), (4,1), (5,4), (1,2), (1,3), (3,2)]
#     g = Graph(links)
    
#     dt = np.linspace(-5, 5, 1000)
#     dw = map( weightUpdate, dt  )
#     plt.plot( dt, dw )
#     plt.show()
        
tests()