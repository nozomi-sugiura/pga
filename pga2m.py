import numpy as np
import math
import sig_inv as si
import pga_lib as pg
from multiprocessing import Pool, cpu_count
import os, sys
from os import getpid, getppid
from scipy.linalg import expm
import scipy as sc
if __name__ == "__main__":
 ntr = 390
 ntr1 = 100
 ntr2 = 1

 d = 4
 n = 4
 mode = 1

 arr = np.load('PCA_n=4.npz')
 g  = arr['arr_0']
 gm_pca = arr['arr_1']
 wp  = arr['arr_2']
 vp  = arr['arr_3']
 lpp0 = gm_pca
 ff = 2
 l = 12
 l2 = ff*l
 sigma = 0.55
 X   = si.sig2path(lpp0 ,l2,d,n,sigma) 
 gm_pca_d = si.tosig(X,n)
 gm_pca_d_i = si.invsig(gm_pca_d,d,n)
 print("s2p and then p2s",0)
 lpp1 = gm_pca+vp[:,mode]
 X   = si.sig2path(lpp1 ,l2,d,n,sigma) 
 vp1_d = si.tosig(X,n)
 v1 = si.logsig(si.prodsig(gm_pca_d_i,vp1_d,d,n),d,n)
 print("s2p and then p2s",1)
 npath = g.shape[1]
 tt = np.zeros(npath)
 for jp in range(npath):
     tt[jp]  = np.sum(vp[:,mode]*(g[:,jp]-gm_pca[:]))
 
 arr = np.load('tPCA_n=4.npz')
 g=arr['arr_0']
 gm=arr['arr_1']
 w=arr['arr_2']
 v=arr['arr_3']
 cov=arr['arr_4']
 sigdim = g.shape[0]
 npath = g.shape[1]

 arr = np.load('PGA1_n=4_j=299.npz')
 tt0=arr['arr_0']
 v0=arr['arr_1']
 

 g0 = np.zeros((sigdim,npath))
 for jp in range(npath):
     ew       = si.expsig(tt0[jp]*v0,d,n)
     g0[:,jp] = si.prodsig(gm,ew,d,n)
 cov_i = np.linalg.pinv(cov)
 P, P_i = pg.cov2P(cov)

 print("#npath", npath)
 eps1 = 0.1 #n=4
 eps2 = 1e-10 #n=4

# v1 = v[:,mode]
# tt = np.zeros(npath)
 nnp = cpu_count()
 print("#main pid={} cpu_count={}".format(getpid(), nnp))
# gm_i = si.invsig(gm,d,n)
# for jp in range(npath):
#     gd     = si.prodsig(gm_i,g[:,jp],d,n)
#     tt[jp] = np.sum(v1[:]*si.logsig(gd,d,n))
 a0 = np.sqrt(np.mean(tt**2))
 v1_old = v1.copy()
 rho = 1
 tk = 1
 gr = np.zeros_like(v1)
 dd = np.zeros_like(v1)
 p = Pool(nnp)
 for j in range(ntr):
   print("#j=",j)
   ttl = tt.tolist()
   values = [(ttl[jp],v1,g,g0,d,n,jp,a0,ntr1,eps1) for jp in range(npath)]
   result = p.map(pg.get_t_wrapper,values)
   tt = np.array(result)
   a1 = np.sqrt(np.mean(tt**2))
   print("#a",a0,a1,a0/a1)
   a0 = np.sqrt(np.mean(tt**2))
   for jp in range(npath):
       print("#inner",jp,tt[jp])#,tt_ini[jp])
   norm_old = 1e20
   for i in range(ntr2):
     v1_new, rho, eps2 = pg.nesterov_ls(v1,v1_old,rho,eps2,tt,g,g0,\
                                        npath,cov,d,n,nnp,P,P_i,cov_i,p)
     v1_old  = v1.copy()
     v1      = v1_new.copy()
   if (j+1)%50 == 0:
     t1  = np.mean(tt**2)
     print("#t1",np.sqrt(np.var(tt)),np.sqrt(t1))
     lp0 = si.prodsig(gm,si.expsig(+np.sqrt(w[mode])*v[:,mode],d,n),d,n)
     ln0 = si.prodsig(gm,si.expsig(-np.sqrt(w[mode])*v[:,mode],d,n),d,n)
     lp1 = si.prodsig(gm,si.expsig(+np.sqrt(t1)*v1,d,n),d,n)
     ln1 = si.prodsig(gm,si.expsig(-np.sqrt(t1)*v1,d,n),d,n)
     fname1 = 'lm2_n=4_j=' + str(j)
     np.savez(fname1, gm, lp0,ln0, lp1, ln1)
     fname2 = 'PGA2_n=4_j=' + str(j)
     np.savez(fname2, tt, v1, tt0, v0)
 for k in range(sigdim):  
   print(k,v1[k],v[k,mode])
 sys.exit()
