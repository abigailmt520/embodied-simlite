import os, sys, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE)); sys.path.insert(0, _HERE)
from sklearn.metrics import roc_auc_score
# we re-run the construction at several (clearance, noise) and report the trade-off:
#   oracle separates iff honest crossing-frac.max < 0.5 < fault crossing-frac.min
#   marginal-clean iff max single-feature AUC <= 0.65   (focus on ||r|| which carries the crossing)
import relational_oracle as RO
from relational_oracle import dist_point_aabb
XI=0.5; D=0.2; WALLC=30.1; A_EDGE,B_EDGE=WALLC-D/2,WALLC+D/2; N=60; AR=0.85
WALL=[(A_EDGE,B_EDGE,9.0,16.0)]
def iso(rng,s):
    e=np.zeros((N,2))
    for i in range(1,N): e[i]=AR*e[i-1]+rng.normal(0,s,2)
    nr=np.linalg.norm(e,axis=1); o=nr>XI; e[o]=e[o]*(XI/nr[o])[:,None]; return e
def truth(rng,side,c):
    y=9.5+6*np.arange(N)/(N-1); j=np.zeros(N)
    for i in range(1,N): j[i]=0.9*j[i-1]+rng.normal(0,0.01)
    x=(A_EDGE-c)-np.abs(j) if side=="A" else (B_EDGE+c)+np.abs(j)
    return np.column_stack([x,y])
def honest(seed,c,s):
    rng=np.random.default_rng(seed); side="A" if seed%2==0 else "B"; t=truth(rng,side,c); return t,t+iso(rng,s)
def fault(seed,c,s):
    rng=np.random.default_rng(100000+seed); side="A" if seed%2==0 else "B"; t=truth(rng,side,c); ob=t.copy()
    if side=="A": ob[:,0]=(B_EDGE+c)+((A_EDGE-c)-t[:,0])
    else: ob[:,0]=(A_EDGE-c)-(t[:,0]-(B_EDGE+c))
    return t,ob+iso(rng,s)
def frac(t,o): return RO.relational_oracle(t,o,WALL,0.0,persist_frac=0.0)["frac"]
def rnorm(t,o): return np.linalg.norm(o-t,axis=1)
print(f"{'c':>5}{'s':>6}{'hon_frac(mean/max)':>20}{'flt_frac(mean/min)':>20}{'oracle_sep?':>12}{'||r||AUC':>10}{'rx_AUC':>9}{'clean?':>8}")
for c in (0.05,0.10,0.15,0.20,0.30):
    for s in (0.15,0.22):
        H=[honest(7000+i,c,s) for i in range(80)]; F=[fault(9000+i,c,s) for i in range(30)]
        hf=np.array([frac(*p) for p in H]); ff=np.array([frac(*p) for p in F])
        hr=np.concatenate([rnorm(*p) for p in H]); fr=np.concatenate([rnorm(*p) for p in F])
        hx=np.concatenate([(o-t)[:,0] for t,o in H]); fx=np.concatenate([(o-t)[:,0] for t,o in F])
        yb=np.r_[np.zeros(len(hr)),np.ones(len(fr))]
        auc_r=max(roc_auc_score(yb,np.r_[hr,fr]),1-roc_auc_score(yb,np.r_[hr,fr]))
        auc_x=max(roc_auc_score(yb,np.r_[hx,fx]),1-roc_auc_score(yb,np.r_[hx,fx]))
        osep = hf.max()<0.5<ff.min()
        clean = max(auc_r,auc_x)<=0.65
        print(f"{c:>5.2f}{s:>6.2f}{f'{hf.mean():.2f}/{hf.max():.2f}':>20}{f'{ff.mean():.2f}/{ff.min():.2f}':>20}{str(osep):>12}{auc_r:>10.3f}{auc_x:>9.3f}{('YES' if (osep and clean) else 'no'):>8}")
print("\nWANT: a row with oracle_sep?=True AND clean?(both AUC<=0.65) -> that is a usable construction.")
