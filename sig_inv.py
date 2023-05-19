#!/usr/bin/env python3
#library for inversion and mean of signature
import numpy as np
#import cupy as cp
#from cupyx import scipy
import scipy.optimize as optimize
import math, sys
from itertools import product
def group_mean(g,d,n,npath,ntr):
  sigdim = (d**(n+1)-1)//(d-1)
  #firstguess
  logp = np.zeros((sigdim,npath))
  for jp in range(npath):
    logp[:,jp] = logsig(g[:,jp],d,n)
  gm = expsig(np.mean(logp,axis=1),d,n)
  #iterations
  for itr in range(ntr):
    gm_i = invsig(gm,d,n)
    for jp in range(npath):
      logp[:,jp]= logsig(prodsig(gm_i,g[:,jp],d,n),d,n)
    gm = prodsig(gm,expsig(np.mean(logp,axis=1),d,n),d,n)
  return gm  
def expsig(x,d,n):
  #Exponential of graded Tensor
  expxt = np.zeros_like(x); expxt[0] = 1.0
  expx  = expxt.copy()
  for i in range(1,n+1):
    expxt =  prodsig(expxt,x,d,n)
    expx  += expxt/math.factorial(i)
  return expx    
def adj_expsig(expx_ad,x,d,n):
  #gradient of the exponential of graded Tensor
  expxt = np.zeros_like(x); expxt[0] = 1.0
  expx  = expxt.copy()
  expxts = np.zeros((expxt.size,n+1))
  for i in range(1,n+1):
    expxts[:,i] = expxt.copy()
    expxt =  prodsig(expxt,x,d,n)
    expx  += expxt/math.factorial(i)
  expxt_ad = np.zeros_like(x)
  x_ad = np.zeros_like(x)
  xt_ad = np.zeros_like(x)
  for i in reversed(range(1,n+1)):
    expxt_ad += expx_ad/math.factorial(i)
    expxt_ad, xt_ad = adj_prodsig(expxt_ad,expxts[:,i],x,d,n)
    x_ad += xt_ad
  x_ad[0] = 0.0    
  return x_ad    
def logsig(x,d,n):
  #Logarithm of graded Tensor
  xm1 = x.copy(); xm1[0] = 0.0
  logx  = np.zeros_like(x)
  logxt = np.zeros_like(x); logxt[0] = -1.0
  for i in range(1,n+1):
    logxt =  prodsig(logxt,-xm1,d,n)
    logx  += logxt/i
  return logx
def invsig(x,d,n):
  #Inverse of graded Tensor
  xm1 = x.copy(); xm1[0] = 0.0
  invxt = np.zeros_like(x); invxt[0] = 1.0
  invx  = invxt.copy()
  for i in range(1,n+1):
    invxt =  prodsig(invxt,-xm1,d,n)
    invx  += invxt
  return invx
#possibly incomplete
def adj_invsig(invx_ad,x,d,n):
  xm1 = x.copy(); xm1[0] = 0.0
  invxt = np.zeros_like(x); invxt[0] = 1.0
  invx  = invxt.copy()
  invxts = np.zeros((invxt.size,n+1))
  for i in range(1,n+1):
    invxts[:,i] = invxt.copy()
    invxt =  prodsig(invxt,-xm1,d,n)
    invx  += invxt
  invxt_ad = np.zeros_like(x)
  xm1_ad = np.zeros_like(xm1)
  a_ad = np.zeros_like(xm1)
  for i in reversed(range(1,n+1)):
    invxt_ad += invx_ad
    invxt_ad, a_ad = adj_prodsig(invxt_ad,invxts[:,i],-xm1,d,n)
    xm1_ad += -a_ad
    a_ad = 0.0
  xm1_ad[0] = 0.0
  return xm1_ad
#possibly incomplete
def adj_logsig(logx_ad,x,d,n):
  xm1 = x.copy(); xm1[0] = 0.0
  logxt = np.zeros_like(x); logxt[0] = -1.0
  logx  = np.zeros_like(x)
  logxts = np.zeros((logxt.size,n+1))
  for i in range(1,n+1):
    logxts[:,i] = logxt.copy()
    logxt =  prodsig(logxt,-xm1,d,n)
    logx  += logxt/i
  logxt_ad = np.zeros_like(x)
  xm1_ad = np.zeros_like(xm1)
  a_ad = np.zeros_like(xm1)
  for i in reversed(range(1,n+1)):
    logxt_ad += logx_ad/i
    logxt_ad, a_ad = adj_prodsig(logxt_ad,logxts[:,i],-xm1,d,n)
    xm1_ad += -a_ad
    a_ad = 0.0
  xm1_ad[0] = 0.0
  return xm1_ad
def quadratic_norm(S,d,n): 
  norm = 0.0
  for k in range(1,n+1):
    j0 = (d**k-1)//(d-1)
    j1 = (d**(k+1)-1)//(d-1)
    Sk = np.sum(S[j0:j1]**2)
    norm += Sk
  return norm            
def adj_quadratic_norm(norm_ad,S_ad,S,d,n):
  for k in reaversed(range(1,n+1)): #bugfix 20230129
    j0 = (d**k-1)//(d-1)
    j1 = (d**(k+1)-1)//(d-1)
    S_ad[j0:j1] += 2*S[j0:j1]*norm_ad
  return S_ad            
def homogenous_norm(S,d,n):
  norm = 0.0
  fn = 1
  eps = sys.float_info.min
  for k in range(1,n+1):
    fk = math.factorial(k)
    j0 = (d**k-1)//(d-1)
    j1 = (d**(k+1)-1)//(d-1)
    Sk = (fk*(np.sum(S[j0:j1]**2)+eps))**(1.0/k)
    norm += Sk*fn
  return norm
def adj_homogenous_norm(norm_ad,S_ad,S,d,n):
  Sk_ad = 0.0
  fn = 1
  eps = sys.float_info.min
  for k in reversed(range(1,n+1)):  #bugfix 20230129
    fk = math.factorial(k)
    j0 = (d**k-1)//(d-1)
    j1 = (d**(k+1)-1)//(d-1)
    Sk = fk*np.sum(S[j0:j1]**2)
    Sk_ad += norm_ad*fn
    Sk_ad *= (1.0/k)*(Sk+eps)**(1.0/k-1)
    S_ad[j0:j1] += 2*Sk_ad*S[j0:j1]*fk
    Sk_ad = 0.0  
  return S_ad            
def cost(u,sigma,vi,Ss,n,m,d):
    u = np.reshape(u,(m,d))
    v = np.zeros_like(u)
    for j in range(m):
        v[j,:] = u[j,:]*sigma[:]+vi[j,:]
#    S = tosig_approx(v,n,1)
    S = tosig(v,n)
    J = 0.0
    Jo = 0.5*homogenous_norm(S-Ss,d,n)
#    Jo = 0.5*quadratic_norm(S-Ss,d,n)
    J += Jo
    J  += 0.5*np.sum(u**2)
    sigma_new = np.sqrt(np.mean(u**2)/(2*Jo/S.size))
#    print("#J,Jo = ",J,Jo,sigma_new)
    return J
def adj_cost(u,sigma,vi,Ss,n,m,d):
    sigdim = Ss.shape[0]
    u = np.reshape(u,(m,d))
    u_ad = np.zeros_like(u)
    v = np.zeros_like(vi)
    S_ad = np.zeros(sigdim)
    for j in range(m):
        v[j,:] = u[j,:]*sigma[:]+vi[j,:]
    Jo_ad = 0.0
    J_ad = 1.0    
    S = tosig(v,n)
    u_ad += J_ad*u
    Jo_ad += J_ad
    S_ad = 0.5*adj_homogenous_norm(Jo_ad,S_ad,S-Ss,d,n)
#    S_ad = 0.5*adj_quadratic_norm(Jo_ad,S_ad,S-Ss,d,n)
    Jo_ad = 0.0
    v_ad = adj_tosig(S_ad,v,n)    
    for j in reversed(range(m)):  #bugfix 20230129
        u_ad[j,:] += v_ad[j,:]*sigma[:]
#        u_ad[j,1:] += v_ad[j,1:]*sigma[1:]
    v_ad = 0
    return u_ad.flatten()
def tosig(v,n):
    d = v.shape[1]
    m = v.shape[0]
    for j in range(m):
        if j==0:
            S  = expv(v[j,:],n)
        else:
            ev = expv(v[j,:],n)
            S  = prodsig(S,ev,d,n) #signature
    return S
def tosig_approx(v,n,n2):
    d = v.shape[1]
    m = v.shape[0]
    for j in range(m):
        if j==0:
            S  = expv_approx(v[j,:],n,n2)
        else:
            ev = expv_approx(v[j,:],n,n2)
            S  = prodsig(S,ev,d,n) #signature
    return S
def norm_sig(S,n,d):
    norm = np.zeros(n+1)
    ic0 = 1; ic1 = 1
    for k in range(1,n+1):
        ic1 += d**k
        norm[k] = np.mean(S[ic0:ic1]**2)
        ic0 = ic1
    norm = np.sqrt(norm)
    return norm
def func(x,norm_sig,n):
    func = 0.0
    ic0 = 1; ic1 = 1
    for k in range(1,n+1):
        func += 0.5*(norm_sig[k]*x[0]**k/x[1]-1)**2
    return func
def adj_tosig(S_ad,v,n):
    d = v.shape[1]
    m = v.shape[0]
    Ss = np.zeros((m,S_ad.shape[0]))
    evs = np.zeros_like(Ss)
    v_ad = np.zeros_like(v)
    #restore intermediate values
    for j in range(m):
        if j==0:
            S  = expv(v[j,:],n)
        else:
            ev = expv(v[j,:],n)
            evs[j,:] = ev.copy()
            Ss[j-1,:] = S.copy()
            S  = prodsig(S,ev,d,n) #signature
    #adjoint        
    for j in reversed(range(m)):
        if j==0:
            v_ad[j,:] = adj_expv(v_ad[j,:],S_ad,v[j,:],n)
            S_ad = 0
        else:
            S_ad, ev_ad = adj_prodsig(S_ad,Ss[j-1,:],evs[j,:],d,n)
            v_ad[j,:] = adj_expv(v_ad[j,:],ev_ad,v[j,:],n)
    return v_ad        
def icf(i,k,d):
    ic = 0
    for j in range(k):
        ic += (i[j]+1)*d**j
    return ic
def powerv(v,e,k,d):
    rd = range(d)
    c = 1.0/math.factorial(k)
    for i in product(rd,repeat=k):
        ic = icf(i,k,d)
        e[ic] = c*np.prod(v[list(i)])
    return e      
def adj_powerv(v_ad,e_ad,v,k,d):
    rd = range(d)
    c = 1.0/math.factorial(k)
    for i in product(rd,repeat=k):
        ic = icf(i,k,d)
        for j in range(k):
            i2 = list(i); i2.pop(j)
            v_ad[i[j]] += c*np.prod(v[i2])*e_ad[ic]
    return v_ad       
def expv(v,n):
    d = v.shape[0]
    dim = (d**(n+1)-1)//(d-1)
    e = np.zeros(dim)
    for k in range(0,n+1):
        e = powerv(v,e,k,d)
    return e        
def expv_approx(v,n,n2):
    d = v.shape[0]
    v0 = np.zeros_like(v)
    dim = (d**(n+1)-1)//(d-1)
    e = np.zeros(dim)
    for k in range(0,n2+1):
        e = powerv(v,e,k,d)
    for k in range(n2+1,n+1):
        e = powerv(v0,e,k,d)
    return e        
def adj_expv(v_ad,e_ad,v,n):
    d = v.shape[0]
    for k in reversed(range(n+1)): #debug 20210130
#    for k in range(n+1):
        v_ad = adj_powerv(v_ad,e_ad,v,k,d)
    return v_ad
def prodsig(ev0,ev1,d,n):
    dim = (d**(n+1)-1)//(d-1)
    e = np.zeros(dim)
    for k in range(n+1):
        e = chen(e,ev0,ev1,k,d)
    return e
def chen(e,ev0,ev1,k,d):
    rd = range(d)
    for i in product(rd,repeat=k):
        ic = icf(i,k,d)
        for j in range(k+1):
            ic0 = icf(list(i)[j:],k-j,d)
            ic1 = icf(list(i)[:j],j,d)
            e[ic] += ev0[ic0]*ev1[ic1]
    return e
def adj_chen(e_ad,ev0_ad,ev1_ad,ev0,ev1,k,d):
    rd = range(d)
    for i in product(rd,repeat=k):  
        ic = icf(i,k,d)
        for j in reversed(range(k+1)): #bugfix 20230129
            ic0 = icf(list(i)[j:],k-j,d)
            ic1 = icf(list(i)[:j],j,d)
            ev0_ad[ic0] += e_ad[ic]*ev1[ic1]
            ev1_ad[ic1] += ev0[ic0]*e_ad[ic]
    return e_ad,ev0_ad,ev1_ad
def adj_prodsig(e_ad,ev0,ev1,d,n):
    ev0_ad = np.zeros_like(ev0); ev1_ad = np.zeros_like(ev1)
    for k in reversed(range(n+1)): #bugfix 20230129
        e_ad,ev0_ad,ev1_ad = adj_chen(e_ad,ev0_ad,ev1_ad,ev0,ev1,k,d)
    e_ad = 0    
    return ev0_ad, ev1_ad
def length(v):
    length = 0.0
    for j in range(v.shape[0]):
        length += np.sqrt(np.sum(v[j,:]**2))
    return length
def dilation(S0,lmda,n,d):
    S = S0.copy()
    for k in range(1,n+1):
        j0 = (d**k-1)//(d-1)
        j1 = (d**(k+1)-1)//(d-1)
        S[j0:j1] *= lmda**k
    return S
def sig2lmda(g,n,d):
    Ss = g.copy()
    l = Ss.shape[1]
    S0 = np.mean(Ss,axis=1)
    lmda0 = [1.0,1.0]
    a = norm_sig(S0,n,d)
    if n>1:
        opt = optimize.minimize(func,lmda0,args=(a,n))
        lmda = opt.x
    else:
        lmda = 1.0
    print("#lmda",lmda[0])
    return lmda[0]
def sig2path(g,m,d,n,sig):
    Ss = g.copy()
    u = np.zeros((m,d))
    sigma = np.full(d,sig)
    vi = np.zeros((m,d))
    for j in range(m):
        vi[j,:] = Ss[1:1+d]/m
    u = np.zeros_like(vi) #"optimal value"
    bnds = np.zeros((m,d,2))
    bnds[:,1:,0] = -10.#dP>0
    bnds[:,:,1] = 10.
#    print("#bounds",bnds[0,:,:])
    bnds=bnds.reshape((m*d,2))
    optv = optimize.minimize(cost,u.flatten(),jac=adj_cost,\
                             args=(sigma,vi,Ss,n,m,d),\
#                             method='BFGS',\
#                             method='L-BFGS-B',\
                             method='SLSQP',\
                             options={'disp': False})
    u = np.reshape(optv.x,(m,d))
    v = np.zeros_like(vi)
    for j in range(m):
        v[j,:] = u[j,:]*sigma[:]+vi[j,:]
    return v
def read_argo(file,l,d,npath):
    data1 = np.loadtxt(file)
    path = np.zeros((l,d,npath))
    for jp in range(npath):
        for i in range(l):
            path[i,0,jp] = data1[i  ,   0]/2000.
            path[i,1,jp] = data1[i,2*jp+1]/2.
            path[i,2,jp] = data1[i,2*jp+2]/20.
    return path
def calc_sigma(paths):
    l     = paths.shape[0]-1
    npath = paths.shape[2]
    a0=0.0; a1=0.0; a2=0.0
    b0=0.0; b1=0.0; b2=0.0
    for jp in range(npath):
        b = np.sum((paths[0,:,jp]-paths[-1,:,jp])**2)**0.5
        b0 += 1.0; b1 += b; b2 += b**2
        for j in range(l):
            a = np.sum(np.diff(paths,axis=0)[j,:,jp]**2)**0.5
            a0 += 1.0; a1 += a; a2 += a**2
    sigma = (a2/a0-(a1/a0)**2)/(b2/b0-(b1/b0)**2)
    return np.sqrt(sigma)
class SetIO():
    def __init__(self, filename: str):
        self.filename = filename
    def __enter__(self):
        sys.stdout = _STDLogger(out_file=self.filename)
    def __exit__(self, *args):
        sys.stdout = sys.__stdout__
class _STDLogger():
    def __init__(self, out_file='out.log'):
        self.log = open(out_file, "w+")
    def write(self, message):
        self.log.write(message)
    def flush(self):
        pass
