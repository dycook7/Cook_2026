"""
Dimensionless-parameter and streamline-angle analysis (Fig. S5) for
Cook et al., Kidney360 (2026).

Usage: place this script in the same folder as the data files (or in a parent
folder; it searches subfolders) and run:  python ca_flow_analysis.py
Required data files: toy_model_data.xlsx, patient_model_data.xlsx
Figures and tables are written to an outputs/ folder next to this script.
"""
import os, numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")   # save figures to files without opening windows (no Tk needed)
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---- paths: data are found in this folder/subfolders (then the parent folder) ----
try:
    BASE = Path(__file__).resolve().parent
except NameError:            # e.g., running cell-by-cell in Jupyter
    BASE = Path.cwd()

def find_data(fname):
    for root in (BASE, BASE.parent):
        hits = [p for p in root.rglob(fname) if p.is_file()]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise FileNotFoundError(f"Found more than one {fname}:\n" + "\n".join(map(str, hits)))
    raise FileNotFoundError(f"Could not find {fname} in {BASE}, its subfolders, or the parent folder.")

OUTDIR = BASE / "outputs"

SENS_THRESHOLDS = [3, 4, 5, 6, 7]     # streamline-angle thresholds (deg) for sensitivity analysis
RHO=1060.0; MU=3.5e-3; ANGLE_THRESHOLD=5.0

#  restrict De to ROIs in the bend. 
BEND_ONLY_DE = True
BEND_LABEL   = "Bend"   # value of the Region column that marks bend ROIs

GEOM={
 "3mm_Toy":dict(d=3.0,ba=125.0),"6mm_Toy":dict(d=6.0,ba=125.0),"9mm_Toy":dict(d=9.0,ba=125.0),
 "P96_3mo":dict(d=6.6,ba=133.0),"P96_12mo":dict(d=6.6,ba=132.0),
 "P98_3mo":dict(d=8.2,ba=130.0),"P98_12mo":dict(d=10.3,ba=115.0),
 "P93_3mo":dict(d=8.1,ba=145.0),"P93_12mo":dict(d=6.6,ba=113.0),
 "P104_3mo":dict(d=11.0,ba=125.0),"P104_12mo":dict(d=9.0,ba=115.0),
}
def bend_radius_mm(d,ba):
    defl=np.deg2rad(180.0-ba); return d/(2.0*np.tan(np.clip(defl,1e-3,None)/2.0))



# ---- paper color scheme (matches the R notebook's my_colors / model_colors) ----
# Hex codes from the source palettes:
#   orange1 = NatParksPalettes::Cuyahoga[2]     orange2 = Cuyahoga[1]
#   green1  = NatParksPalettes::IguazuFalls[2]  green2  = IguazuFalls[1]
#   red1    = nationalparkcolors::Saguaro[2]    red2    = nationalparkcolors::ArcticGates[6]
#   blue1   = nationalparkcolors::CraterLake[1] blue2   = CraterLake[5]
#   lakota1-3 = MetBrewer::Lakota[1:3]
MODEL_COLORS = {
    "P93_3mo":   "#FAAE32",  # orange1
    "P93_12mo":  "#E07529",  # orange2
    "P96_3mo":   "#97AD3D",  # green1
    "P96_12mo":  "#415521",  # green2
    "P98_3mo":   "#E45A5A",  # red1
    "P98_12mo":  "#A12A19",  # red2
    "P104_3mo":  "#7DCCD3",  # blue1
    "P104_12mo": "#376597",  # blue2
    "3mm_Toy":   "#04A3BD",  # lakota1
    "6mm_Toy":   "#F0BE3D",  # lakota2
    "9mm_Toy":   "#931E18",  # lakota3
}
MODEL_ORDER = list(MODEL_COLORS)            # legend / plotting order
def model_label(m):                         # "P93_3mo" -> "P93 3mo", "3mm_Toy" -> "3mm"
    return m.replace("_Toy","").replace("_"," ")
TOY_MARKER, PAT_MARKER = "o", "^"

# ---- safe file writing (Windows locks files that are open in another program) ----
def _fallback(path):
    from datetime import datetime
    p=Path(path); return p.with_name(f"{p.stem}_{datetime.now():%H%M%S}{p.suffix}")

def save_fig(fig,name,dpi=150):
    path=Path(OUTDIR)/name
    try:
        fig.savefig(path,dpi=dpi)
    except OSError as e:
        alt=_fallback(path); fig.savefig(alt,dpi=dpi)
        print(f"[warning] couldn't write {path.name} ({e.strerror}); is it open in another "
              f"program? Saved as {alt.name} instead.")

def save_csv(table,name,**kw):
    path=Path(OUTDIR)/name
    try:
        table.to_csv(path,**kw)
    except OSError as e:
        alt=_fallback(path); table.to_csv(alt,**kw)
        print(f"[warning] couldn't write {path.name} ({e.strerror}); is it open in Excel? "
              f"Saved as {alt.name} instead.")

# ----------------------------------------------------------------------------
def load():
    t=pd.read_excel(find_data("toy_model_data.xlsx")); t["is_toy"]=True
    p=pd.read_excel(find_data("patient_model_data.xlsx")); p["is_toy"]=False
    df=pd.concat([t,p],ignore_index=True)
    df["roi"]=df.ModelName+"|"+df.Region+"|"+df.Wall+"|"+df.Position
    df["high"]=(df.Angle_mean>ANGLE_THRESHOLD).astype(int)
    df["d"]=df.ModelName.map(lambda m:GEOM.get(m,{}).get("d",np.nan))
    df["ba"]=df.ModelName.map(lambda m:GEOM.get(m,{}).get("ba",np.nan))
    df["Rc"]=bend_radius_mm(df["d"],df["ba"])
    Q=df.FlowRate*1e-6/60.0
    df["Re"]=RHO*Q/(0.25*np.pi*(df.d*1e-3)*MU)
    df["De"]=df.Re*np.sqrt((df.d*1e-3)/(2*df.Rc*1e-3))
    if BEND_ONLY_DE:
        # Dean number is only meaningful in the curved segment
        df.loc[df.Region.str.lower()!=BEND_LABEL.lower(), "De"]=np.nan
    df["patient"]=df.ModelName.str.replace(r"_(3mo|12mo)$","",regex=True)
    return df

def set_threshold(df,th):
    """Return a copy of df with 'high' recomputed for angle threshold th (deg)."""
    d=df.copy(); d["high"]=(d.Angle_mean>th).astype(int); return d

def logistic(x,x0,k): return 1.0/(1.0+np.exp(-k*(x-x0)))

# ---- refined threshold table ----
def refined_threshold(df):
    rows=[]
    for m,g in df.groupby('ModelName'):
        g=g.sort_values('FlowRate')
        fmin,fmax=g.FlowRate.min(),g.FlowRate.max()
        b=g.groupby('FlowRate').high.mean()
        frac_lo=b.iloc[0]; frac_hi=b.iloc[-1]
        cross=b[b>=0.5]; onset=cross.index.min() if len(cross) else np.nan
        x=g.FlowRate.values.astype(float); y=g.high.values.astype(float)
        x0=np.nan;k=np.nan;fit_ok=False
        try:
            popt,_=curve_fit(logistic,x,y,p0=[np.median(x),0.01],
                             bounds=([fmin-50,1e-4],[fmax+200,1.0]),maxfev=40000)
            x0,k=popt; fit_ok=(fmin<=x0<=fmax)
        except Exception: pass
        rows.append(dict(model=m,is_toy=bool(g.is_toy.iloc[0]),
            flow_min=round(fmin,1),flow_max=round(fmax,1),
            frac_low=round(frac_lo,2),frac_high=round(frac_hi,2),
            onset_flow_mLmin=(round(onset,1) if np.isfinite(onset) else np.nan),
            onset_below_tested=frac_lo>=0.5,
            logi_x0=round(x0,1),logi_k=round(k,4),midpoint_in_range=fit_ok))
    return pd.DataFrame(rows).sort_values(['is_toy','model'])

# ---- THE 5-PANEL SUMMARY FIGURE ----
def fig_main(df,res):
    fig,axs=plt.subplots(1,5,figsize=(26,4.6))
    # A: disturbed fraction vs flow rate, toy vs patient
    ax=axs[0]
    for toy,marker,color,lab in [(True,'o','#2c7fb8','Toy'),(False,'^','#d95f0e','Patient')]:
        sub=df[df.is_toy==toy].copy()
        sub['fb']=pd.cut(sub.FlowRate,np.linspace(30,430,11))
        tag='toy' if toy else 'patient'
        # mean +/- SEM across all pooled ROI x flow-rate measurements in each flow bin
        grp=sub.groupby('fb',observed=True).high.agg(['mean','sem','count'])
        grp.insert(0,'n_models',sub.groupby('fb',observed=True).ModelName.nunique())
        save_csv(grp.rename(columns={'count':'n_measurements'}),f"figS5A_{tag}_values.csv")
        x=[iv.mid for iv in grp.index]
        ax.errorbar(x,grp['mean'],yerr=grp['sem'],marker=marker,ls='--',lw=.5,color=color,label=lab,capsize=2)
    ax.axhline(0.5,ls=':',c='k',lw=.8); ax.set_ylim(0,1)
    ax.set_xlabel('Flow rate (mL/min)', fontsize=12); ax.set_ylabel('Fraction high-angle ROIs', fontsize=12)
    ax.set_title('Disturbed-flow fraction vs. flow rate'); ax.legend(fontsize=10)
    # B: baseline disturbance per model
    ax=axs[1]; r=res.sort_values(['is_toy','frac_low'])
    colors=['#2c7fb8' if t else '#d95f0e' for t in r.is_toy]
    ax.barh(range(len(r)),r.frac_low,color=colors)
    ax.set_yticks(range(len(r))); ax.set_yticklabels(r.model,fontsize=9)
    ax.axvline(0.5,ls=':',c='k',lw=.8); ax.set_xlim(0,1)
    ax.set_xlabel('Fraction high-angle at physiologic flow', fontsize=12)
    ax.set_title('Baseline disturbance by model', fontsize=12)
    
    # C: Dean collapse
    ax=axs[2]; dd=df.dropna(subset=['De'])
    g=(dd.groupby(['ModelName','FlowRate']).agg(frac=('high','mean'),De=('De','mean'),toy=('is_toy','first')).reset_index())
    # g=(dd.groupby(['ModelName','FlowRate']).agg(frac=('high','mean'),De=('De','mean'),toy=('is_toy','first')).reset_index())
    for m,s in g.groupby('ModelName'):
        mk='o' if s.toy.iloc[0] else '^'; c='#2c7fb8' if s.toy.iloc[0] else '#d95f0e'
        ax.plot(s.De,s.frac,mk,ms=5,color=c,alpha=.7)
    ax.set_xscale('log'); ax.set_xlim(10,1000); ax.set_ylim(-0.05,1.05)
    ax.set_xlabel('Dean number, $\\it{De}$', fontsize=12); ax.set_ylabel('Fraction high-angle ROIs', fontsize=12)
    _cscope=' (bend ROIs)' if BEND_ONLY_DE else ''
    ax.set_title(f"Disturbed fraction vs. De{_cscope}", fontsize=12); ax.legend(fontsize=12)
    import matplotlib.lines as mlines
    # legend handles
    toy_leg = mlines.Line2D([], [], color='#2c7fb8', marker='o', linestyle='None',
                             markersize=6, label='Toy')
    pat_leg = mlines.Line2D([], [], color='#d95f0e', marker='^', linestyle='None',
                              markersize=6, label='Patient')
    ax.legend(handles=[toy_leg, pat_leg], loc='upper right', fontsize=10)

    # D: mean +/- SEM of disturbed fraction vs binned De, toy vs patient.
    # Uses bend-only points (dd = df.dropna(subset=['De']),
    # and De is NaN outside the bend when BEND_ONLY_DE=True). Error bar is the
    # standard error of the per-model-per-flow fractions within each De bin.
    ax=axs[3]
    gd=(dd.groupby(['ModelName','FlowRate'])
          .agg(frac=('high','mean'),De=('De','mean'),toy=('is_toy','first')).reset_index())
    logbins=np.logspace(np.log10(gd.De.min()),np.log10(gd.De.max()),8)
    gd['db']=pd.cut(gd.De,logbins)
    for toy,color,lab in [(True,'#2c7fb8','Toy'),(False,'#d95f0e','Patient')]:
        sub=gd[gd.toy==toy]
        grp=sub.groupby('db',observed=True).frac.agg(['mean','sem','count'])
        xc=[iv.mid for iv in grp.index]
        ax.errorbar(xc,grp['mean'],yerr=grp['sem'],marker='o',ls='--',lw=.5,color=color,
                    label=lab,capsize=3)
    ax.set_xscale('log');  ax.set_xlim(10,1000); ax.set_ylim(0,1)
    ax.axhline(0.5,ls=':',c='k',lw=.8)
    ax.set_xlabel('Dean number, De', fontsize=12); ax.set_ylabel('Fraction high-angle ROIs', fontsize=12)
    ax.set_title(f'Mean \u00b1 SEM disturbed fraction vs. De{_cscope}', fontsize=12); ax.legend(fontsize=10)

    
    # E: Reynolds collapse
    ax=axs[4]; dd=df#.dropna(subset=['Re'])
    g=(dd.groupby(['ModelName','FlowRate']).agg(frac=('high','mean'),Re=('Re','mean'),toy=('is_toy','first')).reset_index())
    # g=(dd.groupby(['ModelName','FlowRate']).agg(frac=('high','mean'),De=('De','mean'),toy=('is_toy','first')).reset_index())
    for m,s in g.groupby('ModelName'):
        mk='o' if s.toy.iloc[0] else '^'; c='#2c7fb8' if s.toy.iloc[0] else '#d95f0e'
        ax.plot(s.Re,s.frac,mk,ms=5,color=c,alpha=.7)
    ax.set_xscale('log'); ax.set_xlim(10,1000); ax.set_ylim(-0.05,1.05)
    ax.set_xlabel('Reynolds number, $\\it{Re}$', fontsize=12); ax.set_ylabel('Fraction high-angle ROIs', fontsize=12)
    ax.set_title(f"Disturbed fraction vs. Re", fontsize=12); ax.legend(fontsize=12)
    import matplotlib.lines as mlines
    # legend handles
    toy_leg = mlines.Line2D([], [], color='#2c7fb8', marker='o', linestyle='None',
                             markersize=6, label='Toy')
    pat_leg = mlines.Line2D([], [], color='#d95f0e', marker='^', linestyle='None',
                              markersize=6, label='Patient')
    ax.legend(handles=[toy_leg, pat_leg], loc='lower left', fontsize=10)
    
    fig.tight_layout(); save_fig(fig,"fig_main.png")
    plt.close("all")

# ---- detail figures ----
def fig_collapse_Re(df):
    # Re is defined everywhere (no curvature term), unlike De (bend only)
    dd=df.dropna(subset=["Re"]).copy(); models=list(dict.fromkeys(dd.ModelName))
    # cmap=plt.get_cmap("tab10"); col={m:cmap(i%10) for i,m in enumerate(models)}
    fig,(a,b)=plt.subplots(1,2,figsize=(12,4.8))

    # Panel A: faded raw ROI scatter, with binned mean +/- SEM overlaid for toy & patient.
    for m in models:
        s=dd[dd.ModelName==m]; mk="o" if s.is_toy.iloc[0] else "^"
        col='#2c7fb8' if s.is_toy.iloc[0] else '#d95f0e'
        a.scatter(s.Re,s.Angle_mean,s=7,alpha=.15,marker=mk,color=col,label=m,zorder=1)
    bins=np.linspace(dd.Re.min(),dd.Re.max(),9)
    dd["_rb"]=pd.cut(dd.Re,bins)
    
    for toy,color,lab in [(True,'#1a1a1a','Toy mean'),(False,'#b30000','Patient mean')]:
        sub=dd[dd.is_toy==toy]
        grp=sub.groupby("_rb",observed=True).Angle_mean.agg(['mean','sem'])
        xc=[iv.mid for iv in grp.index]
        a.errorbar(xc,grp['mean'],yerr=grp['sem'],fmt='o-',ms=4,lw=0.5,capsize=2,
                   color=color,label=lab,zorder=3)
        
    a.axhline(ANGLE_THRESHOLD,ls="--",c="k",lw=.8); a.set_xscale("log"); a.set_xlim(10,1000);
    a.set_xlabel("Reynolds number, $\\it{Re}$", fontsize=12)
    a.set_ylabel("Streamline angle (deg)", fontsize=12)
    a.set_title("Streamline angle vs. $\\it{Re}$"); a.legend(fontsize=8,ncol=1, frameon=False)

    # Panel B: disturbed fraction per model per flow condition, with binomial SEM
    # sqrt(p(1-p)/n) where n = number of ROIs contributing to that point.
    g=(dd.groupby(["ModelName","FlowRate"])
         .agg(frac=("high","mean"),n=("high","size"),Re=("Re","mean"),
              toy=("is_toy","first")).reset_index())
    g["semB"]=np.sqrt(g.frac*(1-g.frac)/g.n)
    for m in models:
        s=g[g.ModelName==m].sort_values("Re"); mk="o" if s.toy.iloc[0] else "^";col='#2c7fb8' if s.toy.iloc[0] else '#d95f0e'
        b.errorbar(s.Re,s.frac,yerr=s["semB"],fmt=mk+"--",ms=4,lw=0.3,capsize=1,
                   color=col,alpha=.8,label=m); b.set_xscale("log"); b.set_xlim(10,1000);
    b.set_xlabel("Reynolds number, $\\it{Re}$", fontsize=12)
    b.set_ylabel("Fraction high-angle ROIs", fontsize=12); b.set_title("Disturbed fraction vs. $\\it{Re}$")
    b.legend(fontsize=8,ncol=1, frameon=False, loc='upper left')
    plt.tight_layout() 
    save_fig(plt.gcf(),"fig_collapse_Re.png")
    plt.close("all")

# ---

def fig_collapse_De(df):
    # Re is defined everywhere (no curvature term), unlike De (bend only)
    dd=df.dropna(subset=["De"]).copy(); models=list(dict.fromkeys(dd.ModelName))
    fig,(a,b)=plt.subplots(1,2,figsize=(12,4.8))

    # Panel A: faded raw ROI scatter, with binned mean +/- SEM overlaid for toy & patient.
    for m in models:
        s=dd[dd.ModelName==m]; mk="o" if s.is_toy.iloc[0] else "^"
        col='#2c7fb8' if s.is_toy.iloc[0] else '#d95f0e'
        a.scatter(s.De,s.Angle_mean,s=7,alpha=.15,marker=mk,color=col,label=m,zorder=1)
    bins=np.linspace(dd.De.min(),dd.De.max(),9)
    dd["_rb"]=pd.cut(dd.De,bins)
    
    for toy,color,lab in [(True,'#1a1a1a','Toy mean'),(False,'#b30000','Patient mean')]:
        sub=dd[dd.is_toy==toy]
        grp=sub.groupby("_rb",observed=True).Angle_mean.agg(['mean','sem'])
        xc=[iv.mid for iv in grp.index]
        a.errorbar(xc,grp['mean'],yerr=grp['sem'],fmt='o-',ms=4,lw=0.5,capsize=2,
                   color=color,label=lab,zorder=3)
    a.axhline(ANGLE_THRESHOLD,ls="--",c="k",lw=.8); a.set_xscale("log"); a.set_xlim(10,1000);
    a.set_xlabel("Dean number, $\\it{De}$", fontsize=12)
    a.set_ylabel("Streamline angle (deg)", fontsize=12)
    a.set_title("Streamline angle vs. $\\it{De}$"); a.legend(fontsize=8,ncol=1, frameon=False)

    # Panel B: disturbed fraction per model per flow condition, with binomial SEM
    # sqrt(p(1-p)/n) where n = number of ROIs contributing to that point.
    g=(dd.groupby(["ModelName","FlowRate"])
         .agg(frac=("high","mean"),n=("high","size"),De=("De","mean"),
              toy=("is_toy","first")).reset_index())
    g["semB"]=np.sqrt(g.frac*(1-g.frac)/g.n)
    for m in models:
        s=g[g.ModelName==m].sort_values("De"); mk="o" if s.toy.iloc[0] else "^"
        col='#2c7fb8' if s.toy.iloc[0] else '#d95f0e'
        b.errorbar(s.De,s.frac,yerr=s["semB"],fmt=mk+"--",ms=4,lw=0.3,capsize=1,
                   color=col,alpha=.8,label=m); b.set_xscale("log"); b.set_xlim(10,1000);
    b.set_xlabel("Dean number, $\\it{De}$", fontsize=12)
    b.set_ylabel("Fraction high-angle ROIs", fontsize=12); b.set_title("Disturbed fraction vs. $\\it{De}$")
    b.legend(fontsize=8,ncol=1, frameon=False, loc='upper left')
    plt.tight_layout() 
    save_fig(plt.gcf(),"fig_collapse_De.png")
    plt.close("all")



# ---- combined Fig. S5 (A-F), colored by model ----
def fig_S5_combined(df,res):
    """Final Fig. S5 layout: 3 rows x 2 columns.
    A-B: fig_main A-B; C-D: fig_collapse_Re; E-F: fig_collapse_De.
    Models are colored with the paper palette; toy = circles, patient = triangles.
    Calculations match the individual figures."""
    import matplotlib.lines as mlines
    models=[m for m in MODEL_ORDER if m in set(df.ModelName)]
    is_toy=df.groupby("ModelName").is_toy.first().to_dict()
    mk=lambda m: TOY_MARKER if is_toy[m] else PAT_MARKER
    fig,axs=plt.subplots(3,2,figsize=(12,14))
    (aA,aB),(aC,aD),(aE,aF)=axs

    # A: fraction high-angle vs flow rate, toy vs patient (same calculation as fig_main A):
    #    mean +/- SEM across all pooled ROI x flow-rate measurements in each flow bin
    bins=np.linspace(30,430,11)
    d=df.copy(); d["fb"]=pd.cut(d.FlowRate,bins)
    for toy,fill in [(True,"white"),(False,"black")]:
        sub=d[d.is_toy==toy].groupby("fb",observed=True).high.agg(["mean","sem"])
        aA.errorbar([iv.mid for iv in sub.index],sub["mean"],yerr=sub["sem"],
                    marker=TOY_MARKER if toy else PAT_MARKER,mfc=fill,mec="black",
                    color="black",ms=7,lw=1.2,capsize=3,zorder=5,
                    label="Toy" if toy else "Patient")
    aA.legend(fontsize=10,frameon=False)
    aA.axhline(.5,ls=":",c="k",lw=.8); aA.set_ylim(0,1)
    aA.set_xlabel("Flow rate (mL/min)",fontsize=12); aA.set_ylabel("Fraction high-angle ROIs",fontsize=12)

    # B: baseline disturbance (lowest flow rate) per model; toy bars hatched.
    r=res.sort_values(["is_toy","frac_low"])
    aB.barh(range(len(r)),r.frac_low,color=[MODEL_COLORS[m] for m in r.model],
            hatch=["//" if t else "" for t in r.is_toy],edgecolor="black",lw=.5)
    aB.set_yticks(range(len(r))); aB.set_yticklabels([model_label(m) for m in r.model],fontsize=10)
    aB.axvline(.5,ls=":",c="k",lw=.8); aB.set_xlim(0,1)
    aB.set_xlabel("Fraction high-angle at physiologic flow",fontsize=12)

    # C/E: streamline angle vs Re/De (faded ROI x flow points; black = binned means)
    # D/F: fraction high-angle per model per flow rate, binomial SE (as in fig_collapse_*)
    for xcol,aL,aR,lab in [("Re",aC,aD,"Reynolds number, $\\it{Re}$"),
                           ("De",aE,aF,"Dean number, $\\it{De}$")]:
        dd=df.dropna(subset=[xcol]).copy()
        for m in models:
            s=dd[dd.ModelName==m]
            aL.scatter(s[xcol],s.Angle_mean,s=8,alpha=.25,marker=mk(m),color=MODEL_COLORS[m],zorder=1)
        dd["_b"]=pd.cut(dd[xcol],np.linspace(dd[xcol].min(),dd[xcol].max(),9))
        for toy,fill in [(True,"white"),(False,"black")]:
            grp=dd[dd.is_toy==toy].groupby("_b",observed=True).Angle_mean.agg(["mean","sem"])
            aL.errorbar([iv.mid for iv in grp.index],grp["mean"],yerr=grp["sem"],
                        marker=TOY_MARKER if toy else PAT_MARKER,mfc=fill,mec="black",
                        color="black",ms=6,lw=.8,capsize=2,zorder=3)
        aL.axhline(ANGLE_THRESHOLD,ls="--",c="k",lw=.8)
        aL.set_xscale("log"); aL.set_xlim(10,1000)
        aL.set_xlabel(lab,fontsize=12); aL.set_ylabel("Streamline angle (deg)",fontsize=12)

        g=(dd.groupby(["ModelName","FlowRate"])
             .agg(frac=("high","mean"),n=("high","size"),x=(xcol,"mean")).reset_index())
        g["se"]=np.sqrt(g.frac*(1-g.frac)/g.n)
        for m in models:
            s=g[g.ModelName==m].sort_values("x")
            aR.errorbar(s.x,s.frac,yerr=s.se,fmt=mk(m)+"--",ms=5,lw=.4,capsize=1,
                        color=MODEL_COLORS[m],alpha=.9)
        aR.axhline(.5,ls=":",c="k",lw=.8)
        aR.set_xscale("log"); aR.set_xlim(10,1000); aR.set_ylim(-0.05,1.05)
        aR.set_xlabel(lab,fontsize=12); aR.set_ylabel("Fraction high-angle ROIs",fontsize=12)

    for ax,letter in zip(axs.flat,"ABCDEF"):
        ax.text(-0.14,1.04,letter,transform=ax.transAxes,fontsize=16,fontweight="bold",va="bottom")

    # shared legend: models (color + shape) and group means
    handles=[mlines.Line2D([],[],color=MODEL_COLORS[m],marker=mk(m),ls="None",ms=7,label=model_label(m))
             for m in models]
    handles+=[mlines.Line2D([],[],color="black",marker=TOY_MARKER,mfc="white",ls="None",ms=7,label="Toy mean"),
              mlines.Line2D([],[],color="black",marker=PAT_MARKER,ls="None",ms=7,label="Patient mean")]
    fig.legend(handles=handles,loc="lower center",ncol=7,fontsize=10,frameon=False,bbox_to_anchor=(.5,0))
    fig.tight_layout(rect=(0,.05,1,1))
    save_fig(fig,"fig_S5_combined.png",dpi=300)
    save_fig(fig,"fig_S5_combined.pdf")       # vector version for submission
    plt.close("all")

# ---- angle noise floor (rationale for the 5-degree threshold) ----
def noise_floor(df):
    """Streamline-angle distribution in known-laminar flow: prebend of the 3- and
    6-mm toy models, which remained laminar at all flow rates."""
    ref=df[df.ModelName.isin(["3mm_Toy","6mm_Toy"])&(df.Region=="Prebend")]
    q=ref.Angle_mean.quantile([.5,.9,.95,.99])
    out=pd.DataFrame({"percentile":[50,90,95,99],"angle_deg":q.values.round(2)})
    out["n_obs"]=len(ref)
    save_csv(out,f"angle_noise_floor.csv",index=False)
    print("\n=== Noise floor: streamline angle in laminar toy prebend (3+6 mm) ===")
    print(out.to_string(index=False))
    return out

# ---- angle-threshold sensitivity (3-7 degrees) ----
def sensitivity(df,thresholds=SENS_THRESHOLDS):
    from scipy.stats import mannwhitneyu
    bins=np.linspace(30,430,11); per_model=[]; summary=[]
    for th in thresholds:
        d=set_threshold(df,th)
        for m,g in d.groupby("ModelName"):
            b=g.groupby("FlowRate").high.mean(); cross=b[b>=0.5]
            per_model.append(dict(threshold=th,model=m,is_toy=bool(g.is_toy.iloc[0]),
                frac_lowest_flow=round(b.iloc[0],2),frac_highest_flow=round(b.iloc[-1],2),
                overall_frac=round(g.high.mean(),2),
                onset_flow_mLmin=round(cross.index.min(),1) if len(cross) else np.nan))
        pm=pd.DataFrame([r for r in per_model if r["threshold"]==th])
        toy,pat=pm[pm.is_toy],pm[~pm.is_toy]
        d["fb"]=pd.cut(d.FlowRate,bins)
        mb=(d.groupby(["ModelName","fb"],observed=True)
              .agg(f=("high","mean"),toy=("is_toy","first")).reset_index()
              .groupby(["fb","toy"],observed=True).f.mean().unstack())
        both=mb.dropna()
        summary.append(dict(threshold=th,
            toy_mean_overall_frac=round(toy.overall_frac.mean(),2),
            patient_mean_overall_frac=round(pat.overall_frac.mean(),2),
            every_patient_model_above_every_toy=bool(pat.overall_frac.min()>toy.overall_frac.max()),
            mannwhitney_p_model_level=round(mannwhitneyu(pat.overall_frac,toy.overall_frac,
                                               alternative="greater",method="exact").pvalue,4),
            flow_bins_patient_gt_toy=f"{int((both[False]>both[True]).sum())}/{len(both)}",
            patient_models_majority_disturbed_at_lowest=f"{int((pat.frac_lowest_flow>=0.5).sum())}/{len(pat)}",
            toy_models_majority_disturbed_at_lowest=f"{int((toy.frac_lowest_flow>=0.5).sum())}/{len(toy)}",
            toy_models_reaching_50pct=", ".join(toy[toy.onset_flow_mLmin.notna()].model) or "none"))
    per_model=pd.DataFrame(per_model); summary=pd.DataFrame(summary)
    save_csv(per_model,f"sensitivity_per_model.csv",index=False)
    save_csv(summary,f"sensitivity_summary.csv",index=False)
    print("\n=== Threshold sensitivity (model-level) ===")
    print(summary.to_string(index=False))

    # figure: Fig S5A-style curves (model-level mean +/- SEM) at each threshold
    fig,axs=plt.subplots(1,len(thresholds),figsize=(4*len(thresholds),3.8),sharey=True)
    for ax,th in zip(np.atleast_1d(axs),thresholds):
        d=set_threshold(df,th); d["fb"]=pd.cut(d.FlowRate,bins)
        for toy,marker,color,lab in [(True,'o','#2c7fb8','Toy'),(False,'^','#d95f0e','Patient')]:
            pmb=d[d.is_toy==toy].groupby(["ModelName","fb"],observed=True).high.mean().reset_index()
            grp=pmb.groupby("fb",observed=True).high.agg(["mean","sem"])
            ax.errorbar([iv.mid for iv in grp.index],grp["mean"],yerr=grp["sem"],marker=marker,
                        ls="--",lw=.5,color=color,label=lab,capsize=2)
        ax.axhline(.5,ls=":",c="k",lw=.8); ax.set_ylim(0,1)
        ax.set_title(f"Threshold = {th}\u00b0"); ax.set_xlabel("Flow rate (mL/min)")
    np.atleast_1d(axs)[0].set_ylabel("Fraction high-angle ROIs"); np.atleast_1d(axs)[0].legend(fontsize=9)
    fig.tight_layout(); save_fig(fig,"fig_sensitivity_thresholds.png"); plt.close("all")

    # figure: "Baseline disturbance by model" (fraction high-angle at the lowest
    # flow rate) at each threshold. Model order is fixed across tiles (sorted by the
    # 5-degree result, or the middle threshold) so bars can be compared tile to tile.
    ref_th=5 if 5 in thresholds else thresholds[len(thresholds)//2]
    order=(per_model[per_model.threshold==ref_th]
           .sort_values(["is_toy","frac_lowest_flow"])[["model","is_toy"]])
    fig,axs=plt.subplots(1,len(thresholds),figsize=(3.6*len(thresholds),4.6),sharey=True)
    for ax,th in zip(np.atleast_1d(axs),thresholds):
        r=order.merge(per_model[per_model.threshold==th][["model","frac_lowest_flow"]],on="model")
        colors=['#2c7fb8' if t else '#d95f0e' for t in r.is_toy]
        ax.barh(range(len(r)),r.frac_lowest_flow,color=colors)
        ax.axvline(0.5,ls=":",c="k",lw=.8); ax.set_xlim(0,1)
        ax.set_title(f"Threshold = {th}\u00b0"); ax.set_xlabel("Fraction high-angle\nat physiologic flow")
    ax0=np.atleast_1d(axs)[0]
    ax0.set_yticks(range(len(order))); ax0.set_yticklabels(order.model,fontsize=9)
    fig.tight_layout(); save_fig(fig,"fig_sensitivity_baseline.png"); plt.close("all")
    return summary,per_model

def main():
    os.makedirs(OUTDIR,exist_ok=True)
    df=load()
    print(f"Reading: {find_data('toy_model_data.xlsx')}\n         {find_data('patient_model_data.xlsx')}")
    print(f"Writing outputs to: {OUTDIR}")
    print(f"Loaded {len(df)} rows | {df.ModelName.nunique()} models | {df.roi.nunique()} ROIs")
    res=refined_threshold(df)
    # write CSVs
    save_csv(df,f"roi_with_dimensionless.csv",index=False)
    save_csv(res,f"threshold_refined.csv",index=False)
    # figures
    fig_main(df,res)
    fig_collapse_Re(df)
    fig_collapse_De(df)
    fig_S5_combined(df,res)
    # angle-threshold rationale and sensitivity (R1-M2)
    noise_floor(df)
    sensitivity(df)

if __name__=="__main__":
    main()
