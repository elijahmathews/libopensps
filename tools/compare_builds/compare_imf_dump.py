"""Compare two imf_batch_dump files (spectra, magnitudes, mass, L_bol).

Usage: python3 compare_imf_dump.py REFERENCE TEST   (see tools/compare_builds/README.md)
"""
import sys, numpy as np
def load(f):
    d=open(f,'rb').read()
    ns,n,nb=np.frombuffer(d[:12],dtype=np.int32)
    a=np.frombuffer(d[12:],dtype=np.float64)
    o=0
    spec=a[o:o+ns*n].reshape(n,ns); o+=ns*n
    mags=a[o:o+nb*n].reshape(n,nb); o+=nb*n
    mass=a[o:o+n]; o+=n; lbol=a[o:o+n]
    return spec,mags,mass,lbol
r=load(sys.argv[1]); t=load(sys.argv[2])
sp=r[0]; peak=sp.max(axis=1,keepdims=True); m=sp>1e-6*peak
ds=np.abs(t[0]-r[0])[m]/np.abs(r[0])[m]
bright=r[1]<r[1].min(axis=1,keepdims=True)+30
dm=np.abs(t[1]-r[1])[bright]
print(f"spec max rel {ds.max():.2e} (median {np.median(ds):.1e}); mags max {dm.max():.2e}; mass max rel {np.max(np.abs(t[2]/r[2]-1)):.2e}; lbol max rel {np.max(np.abs(t[3]/r[3]-1)):.2e}")
