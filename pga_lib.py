import numpy as np
import sig_inv as si
from scipy.linalg import expm
def cost_wrapper(args):
    return cost(*args)
def cost(tt,vv,g,g0,d,n,npath):
    cost = 0.0
    for jp in range(npath):
        t = tt[jp]
        dL = defDL(t,vv,g0,d,n,jp)
        X  = defX(t,vv,g,g0,d,n,jp)
        dLX = si.prodsig(dL,X,d,n)
        cost += np.dot(dLX,dLX)
    cost /= (2.0*npath)
    cost += np.dot(vv,vv)/2.0
    if np.isnan(cost):
        cost = 1e30
    return cost
def defX(t,v,g,g0,d,n,jp):
    mew   = defDL(t,v,g0,d,n,jp)
    mew_i = si.invsig(mew,d,n)
    aa    = si.prodsig(mew_i,g[:,jp],d,n)
    X     = si.logsig(aa,d,n)
    return X
def defDL(t,v,g0,d,n,jp):
    w     = t*v
    ew    = si.expsig(w,d,n)
    mew   = si.prodsig(g0[:,jp],ew,d,n)
    return mew
def commutator(X,Y,d,n):
    return si.prodsig(X,Y,d,n)-si.prodsig(Y,X,d,n)
def defQ(P,v1,d,n):
    Q = np.zeros_like(P)
    for j in range(P.shape[1]):
      Q[:,j]  = commutator(v1,P[:,j],d,n)
    return Q
def cov2P(cov):
    sigdim = cov.shape[0]
    rnk = np.linalg.matrix_rank(cov)
    ww,vv = np.linalg.eigh(cov)
    idx = ww.argsort()[::-1]   
    ww = ww[idx]; vv = vv[:,idx]
    P = vv[:,:rnk]
    P_i = np.linalg.pinv(P)
    return P, P_i
def grad_wj(t,v,g,g0,d,n,jp,P,P_i,Q,A):
    X     = defX(t,v,g,g0,d,n,jp)
    w_ad  = Jacobi2(P,P_i,Q,A,t,v,d,n,X,g0,jp)
    return w_ad
def defW(P,P_i,Q,A,t):
    W = np.linalg.pinv(P_i @ Q) @ P_i
    m = P.shape[1]
    B = expm(t*A)
    I = np.eye(m)
    W = (I-B) @ W
    W = P @ W
    return W
def defXIout(P,P_i,Q,A,t,v,g0,jp,dL,d,n):
    W = defW(P,P_i,Q,A,t) 
    sigdim = P.shape[0]
    m = P.shape[1]
    XIout = np.zeros((sigdim,sigdim))
    for j in range(sigdim):
        XIout[:,j] = si.prodsig(dL,W[:,j],d,n)
    return XIout
def Jacobi2(P,P_i,Q,A,t,v,d,n,X,g0,jp):
    sigdim = P.shape[0]
    m = P.shape[1]
    dL    = defDL(t,v,g0,d,n,jp)
    XIout = defXIout(P,P_i,Q,A,t,v,g0,jp,dL,d,n)
    dLX    = (-2)*si.prodsig(dL,X,d,n)
    prod_j = XIout.T @ dLX
    Nabla  = prod_j / t   
    return Nabla
def get_t_wrapper(args):
    return get_t(*args)
def grad_t_ini(t,v,g,g0,d,n,jp):
    X  = defX(t,v,g,g0,d,n,jp)
    dL = defDL(t,v,g0,d,n,jp)
    dLX = (-2)*si.prodsig(dL,X,d,n)
    dLv = si.prodsig(dL,v,d,n)
    Nabla = np.dot(dLX, dLv)
    return Nabla
def get_t(t,v1,g,g0,d,n,jp,a0,ntr1,eps1):
    dt_ini = a0
    gr_old = 1e20
    sigdim = g.shape[0]
    npath  = g.shape[1]
    t_old = t
    for i in range(ntr1):
        inc  = grad_t_ini(t,v1,g,g0,d,n,jp)
        t -= inc*eps1
        if jp==0:
            print("#dbg",i,inc*eps1,t,t_old)
        if np.abs(inc*eps1)<1e-2:
            break
    return t
def nesterov_ls(xk,xkm1,rhokm1,eta,tt,g,g0,npath,cov,d,n,nnp,P,P_i,cov_i,p):
    rhok = 0.5*(1+np.sqrt(1+4*rhokm1**2))
    rk   = (rhokm1-1)/rhok
    xkt  = np.zeros_like(xk)
    xkp1 = np.zeros_like(xk)
    c_old = cost(tt,xk,g,g0,d,n,npath)
    xkt  = xk + rk*(xk-xkm1)
    gr  = xkt + 0.5*grad_v(tt,xkt,g,g0,npath,d,n,P,P_i)
    rr = 1.25
    aa = np.zeros(nnp)
    for j in range(nnp):
        aa[j] = eta*rr**j
    values = [(tt,xkt-aa[j]*gr,g,g0,d,n,npath) for j in range(nnp)]
    result = p.map(cost_wrapper,values)
    c = np.array(result)
    ind = c.argsort()
    print("#cost",*c[ind[:3]])
    aa_opt = aa[ind[0]]
    if c_old < c[ind[0]]:
        xkp1 = xk.copy()
    else:
        xkp1 = xkt - aa_opt*gr
    norm0 = si.quadratic_norm(gr,d,n)
    absx0  = si.homogenous_norm(xk,d,n)
    absx1  = si.homogenous_norm(xkp1,d,n)
    lmda = np.sqrt(absx0/absx1)
    xkp1 = si.dilation(xkp1,lmda,n,d)
    absx1  = si.homogenous_norm(xkp1,d,n)
    c = cost(tt,xkp1,g,g0,d,n,npath)
    if ind[0]== 0:
        eta /= rr
    elif ind[0] == nnp-1:
        eta *= rr
    print("#outer", norm0, absx1, c, ind[0])
    return xkp1, rhok, eta
def nesterov_p(xk,xkm1,rhokm1,eta,tt,g,g0,npath,cov,d,n,nnp,P,P_i,cov_i):
    rhok = 0.5*(1+np.sqrt(1+4*rhokm1**2))
    rk   = (rhokm1-1)/rhok
    xkt  = np.zeros_like(xk)
    xkp1 = np.zeros_like(xk)
    xkt  = xk + rk*(xk-xkm1)
    gr  = grad_v(tt,xkt,g,g0,npath,d,n,P,P_i)
    nnp2 = 32
    rr = 1.5
    xkp1 = xkt - eta*rr*gr
    norm0 = si.quadratic_norm(gr,d,n)
    absx0  = si.homogenous_norm(xk,d,n)
    absx1  = si.homogenous_norm(xkp1,d,n)
    lmda = np.sqrt(absx0/absx1)
    xkp1 = si.dilation(xkp1,lmda,n,d)
    absx1  = si.homogenous_norm(xkp1,d,n)
    c = cost(tt,xkp1,g,g0,d,n,npath)
    print("#outer", norm0, absx1, c)
    return xkp1, rhok, eta
def adagrad(xk,h,eta,tt,g,g0,npath,cov,d,n,nnp,P,P_i,cov_i):
    sigdim = g.shape[0]
    gr  = grad_v(tt,xk,g,g0,npath,d,n,P,P_i)
    for i1 in range(sigdim):
      h[i1]  += gr[i1]*gr[i1]
    hsqrt = np.zeros(sigdim)  
    hsqrt = (h+1e-10)**(-0.5)
    inc = np.zeros(sigdim)
    inc = eta * hsqrt * gr
    inc = P @ P_i @ inc
    xkp1 = xk - inc
    norm0 = si.quadratic_norm(gr,d,n)
    absx0  = si.homogenous_norm(xk,d,n)
    absx1  = si.homogenous_norm(xkp1,d,n)
    lmda = np.sqrt(absx0/absx1)
    xkp1 = si.dilation(xkp1,lmda,n,d)
    absx1  = si.homogenous_norm(xkp1,d,n)
    c = cost(tt,xkp1,g,g0,d,n,npath)
    print("#outer", norm0, absx1, c)
    return xkp1, h
def grad_v(tt,v1,g,g0,npath,d,n,P,P_i):
    sigdim = (d**(n+1)-1)//(d-1)
    bb = np.zeros(sigdim)
    Q = defQ(P,v1,d,n)
    A = -P_i @ Q
    npath  = g.shape[1]
    for jp in range(npath):
      t = tt[jp]  
      w = t*v1
      bb += t*grad_wj(t,v1,g,g0,d,n,jp,P,P_i,Q,A)
    bb = bb/npath  
    return bb
