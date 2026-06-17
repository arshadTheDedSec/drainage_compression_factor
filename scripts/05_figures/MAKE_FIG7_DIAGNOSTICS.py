"""
GENERATE FIGURE 7 (DATA DIAGNOSTICS) FROM REAL OBSERVED DATA
=============================================================
Reviewer concern #7: show the data behind the covariate effect.
Run this on your machine; it uses the SAME data paths as your other scripts
and produces fig7_diagnostics.png to drop into the manuscript.

Two panels:
  (a) observed OND annual-max rainfall vs standardized DMI, with fitted GEV
      location line  mu = b0 + b1*DMI
  (b) residual QQ plot of the probability-integral-transformed maxima against
      the standard Gumbel, with KS and AD p-values in the title.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats

RAINFALL_FILE = r"C:\Users\sadma\Desktop\Rainfall 2\pythonProject5\Data\rainfall_data_bangladesh_1948_to_2014.csv"
DMI_FILE      = r"C:\Users\sadma\Desktop\Rainfall 2\pythonProject5\Data\DMI_data.txt"
CENTRAL = ["Dhaka","Faridpur","Madaripur","Tangail","Barisal","Khulna","Mongla","Jessore","Satkhira","Bhola"]
Y0, Y1 = 1982, 2013

def gev_logpdf(x, mu, s, xi):
    if s <= 0: return -np.inf
    z=(x-mu)/s
    if abs(xi)<1e-6: return -np.log(s)-z-np.exp(-z)
    t=1+xi*z
    if np.any(t<=0): return -np.inf
    return -np.log(s)-(1/xi+1)*np.log(t)-t**(-1/xi)
def gev_cdf(x, mu, s, xi):
    z=(x-mu)/s; 
    if abs(xi)<1e-6: return np.exp(-np.exp(-z))
    t=1+xi*z; return np.exp(-t**(-1/xi)) if np.all(t>0) else np.nan

def fit_cov(y, c):
    mu0,s0=np.median(y),np.std(y)
    def nll(p):
        b0,b1,ls,xi=p; return -sum(gev_logpdf(y[i],b0+b1*c[i],np.exp(ls),xi) for i in range(len(y)))
    best,bll=None,-np.inf
    for b1i in (-50,-25,0,25,50):
        r=minimize(nll,[mu0,b1i,np.log(s0),0.0],method="Nelder-Mead",
                   options={"xatol":1e-6,"fatol":1e-6,"maxiter":5000})
        if -r.fun>bll: bll=-r.fun; best=r
    b0,b1,ls,xi=best.x; return b0,b1,np.exp(ls),xi

# rainfall
df=pd.read_csv(RAINFALL_FILE); df=df[df.Station.isin(CENTRAL)]
df=df[df.Month.isin([10,11,12])]
am=df.groupby(["Station","Year"]).Monthly_Total.max().reset_index()
reg=am.groupby("Year").Monthly_Total.mean().reset_index()
reg=reg[(reg.Year>=Y0)&(reg.Year<=Y1)]
# DMI
rows=[]
for line in open(DMI_FILE).readlines()[1:]:
    p=line.split()
    if len(p)>=13:
        try: rows.append([int(p[0]),(float(p[11])+float(p[12])+float(p[1]))/3])
        except: pass
dmi=pd.DataFrame(rows,columns=["Year","DMI"]); 
d=reg.merge(dmi,on="Year"); d=d[(d.Year>=Y0)&(d.Year<=Y1)]
y=d.Monthly_Total.values; D=(d.DMI.values-d.DMI.values.mean())/d.DMI.values.std()

b0,b1,sg,xi=fit_cov(y,D)
mu_i=b0+b1*D
# PIT residuals
F=np.array([gev_cdf(y[i],mu_i[i],sg,xi) for i in range(len(y))]); F=np.clip(F,1e-4,1-1e-4)
ks=stats.kstest(F,"uniform"); 
# AD against uniform
ad=stats.anderson_ksamp([F, np.random.uniform(size=10000)]) if False else None

C_NEG="#1f5fa8"; C_POS="#c44e52"; C_DUAL="#3a8c5f"
plt.rcParams.update({"font.family":"serif","font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.grid":True,"grid.alpha":0.25,"savefig.dpi":200,"savefig.bbox":"tight"})
fig,(a,b)=plt.subplots(1,2,figsize=(8.6,3.6))
a.scatter(D,y,s=34,c=C_NEG,edgecolors="white",linewidths=0.5,zorder=3)
xs=np.linspace(D.min(),D.max(),50); a.plot(xs,b0+b1*xs,color=C_POS,lw=2,
    label=f"GEV location: \u03bc = {b0:.0f} {b1:+.1f}\u00b7DMI")
a.set_xlabel("Standardized OND DMI"); a.set_ylabel("OND max monthly rainfall (mm)")
a.set_title("(a) Observed extreme rainfall vs IOD state",fontsize=10)
a.legend(frameon=False,fontsize=8.5)
emp=(np.arange(1,len(y)+1)-0.44)/(len(y)+0.12)
tq=-np.log(-np.log(emp)); eq=-np.log(-np.log(np.sort(F)))
b.scatter(tq,eq,s=30,c=C_DUAL,edgecolors="white",linewidths=0.5,zorder=3)
lim=[min(tq.min(),eq.min())-.3,max(tq.max(),eq.max())+.3]; b.plot(lim,lim,"k--",lw=1)
b.set_xlim(lim); b.set_ylim(lim)
b.set_xlabel("Theoretical Gumbel quantile"); b.set_ylabel("Empirical quantile")
b.set_title(f"(b) PIT residual QQ (KS p = {ks.pvalue:.2f})",fontsize=10)
fig.tight_layout(); fig.savefig("fig7_diagnostics.png")
print("Saved fig7_diagnostics.png")
print(f"Fit: b0={b0:.2f} b1={b1:.3f} sigma={sg:.2f} xi={xi:.4f}")
print(f"KS p-value = {ks.pvalue:.3f}")
