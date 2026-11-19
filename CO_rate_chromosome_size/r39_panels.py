#!/usr/bin/env python3
"""R3.9 figures: revised Fig. 3b and Supplementary Fig. [MixedModel] (a, b). Run after r39_analysis.py."""
import pandas as pd, numpy as np, json, statsmodels.formula.api as smf, warnings
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
warnings.filterwarnings('ignore')
df=pd.read_csv('CO_rate_per_chromosome.tsv',sep='\t'); mlh=pd.read_csv('MLH1_genomewide_points.tsv',sep='\t'); F=json.load(open('fits.json'))
COL={'R_barbata':'#7570B3','R_breviuscula':'#1B9E77','R_cephalotes':'#D95F02','R_ciliata':'#3399FF','R_colorata':'#66A61E','R_nervosa':'#E6AB02','R_pubera':'#E7298A'}
SP=list(COL); nm=lambda s:s.replace('R_','R. ')
def style(theme):
    FG,MUT,BG=('#000000','#595959','#ffffff') if theme=='white' else ('#ffffff','#bdbdbd','#000000')
    plt.rcParams.update({'font.family':['Arial','Liberation Sans'],'font.size':6,'axes.linewidth':0.5,'xtick.major.width':0.5,'ytick.major.width':0.5,'xtick.minor.width':0.4,'ytick.minor.width':0.4,'xtick.major.size':2.5,'ytick.major.size':2.5,'xtick.minor.size':1.5,'ytick.minor.size':1.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.edgecolor':FG,'xtick.color':FG,'ytick.color':FG,'text.color':FG,'axes.labelcolor':FG})
    return FG,MUT,BG
def logaxes(ax):
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_yticks([0.01,0.02,0.05,0.1,0.2,0.5,1,2]); ax.set_yticklabels(['0.01','0.02','0.05','0.1','0.2','0.5','1','2'])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
def fig3b(theme):
    FG,MUT,BG=style(theme); plt.rcParams.update({'font.size':6.5,'axes.labelsize':7.5}); W,H=169/72,161/72
    fig=plt.figure(figsize=(W,H),facecolor=BG); ax=fig.add_axes([0.19,0.17,0.77,0.78],facecolor=BG)
    for s in SP:
        g=df[df.species==s]; ax.scatter(g['size'],g.co_rate,s=12,color=COL[s],edgecolor=BG,linewidth=0.3,zorder=3)
    x=np.logspace(np.log10(df['size'].min()),np.log10(df['size'].max()),50); a,b,r2=F['ols']
    ax.plot(x,10**(a+b*np.log10(x)),color=FG,lw=0.8,zorder=2)
    for _,r in mlh.iterrows(): ax.scatter(r['size'],r.co_rate,marker='^',s=16,facecolor='none',edgecolor=COL[r.species],linewidth=0.8,zorder=3)
    logaxes(ax); ax.set_xlim(2.3e7,2.2e9); ax.set_ylim(0.0035,3.2)
    ax.set_xlabel('Chromosome size (bp)'); ax.set_ylabel('Crossover rate (cM/Mb)')
    ax.text(0.97,0.97,f'adj. $R^2$ = {r2:.2f}\nLMM slope = −{abs(F["mm"][1]):.2f}\n$P$ = 1.7 × 10$^{{-22}}$',transform=ax.transAxes,ha='right',va='top',fontsize=5.5,linespacing=1.3)
    h=[Line2D([],[],marker='o',ls='',color=COL[s],markeredgecolor=BG,markersize=3.5,label=nm(s)) for s in SP]
    h+=[Line2D([],[],ls='',label=''),Line2D([],[],marker='o',ls='',color=FG,markersize=3,label='Genetic map, per chromosome'),
        Line2D([],[],marker='^',ls='',markerfacecolor='none',markeredgecolor=FG,markersize=3.5,label='MLH1 foci, genome-wide\n(not fitted)')]
    leg=ax.legend(handles=h,loc='lower left',frameon=False,fontsize=5.5,handletextpad=0.1,borderaxespad=0.3,labelspacing=0.15)
    for t in leg.get_texts()[:7]: t.set_fontstyle('italic')
    fig.text(0.01,0.985,'b',fontsize=8,fontweight='bold',va='top')
    o=f'Fig3b_CO_rate_vs_chrsize_{theme}'; fig.savefig(o+'.svg',transparent=True); fig.savefig(o+'.pdf',facecolor=BG); fig.savefig(o+'.png',dpi=600,facecolor=BG); plt.close(fig)
def suppl(theme='white'):
    FG,MUT,BG=style(theme); plt.rcParams.update({'font.size':6,'axes.labelsize':6}); fig=plt.figure(figsize=(120/25.4,58/25.4),facecolor=BG)
    ax=fig.add_axes([0.09,0.20,0.38,0.74],facecolor=BG); bx=fig.add_axes([0.60,0.20,0.38,0.74],facecolor=BG)
    a,b,p=F['mm']
    for s in SP:
        g=df[df.species==s]; ax.scatter(g['size'],g.co_rate,s=10,color=COL[s],edgecolor=BG,linewidth=0.3,zorder=3)
        o=smf.ols('ly~lx',g).fit(); x=np.logspace(g.lx.min(),g.lx.max(),10); ax.plot(x,10**(o.params.Intercept+o.params.lx*np.log10(x)),color=COL[s],lw=0.8,zorder=2)
    x=np.logspace(df.lx.min(),df.lx.max(),50); ax.plot(x,10**(a+b*np.log10(x)),color=FG,lw=0.6,ls='--',zorder=1)
    logaxes(ax); ax.set_xlim(2.3e7,5e8); ax.set_ylim(0.015,3.2); ax.set_xlabel('Chromosome size (bp)'); ax.set_ylabel('Crossover rate (cM/Mb)')
    ax.text(0.97,0.97,f'Mixed model (dashed):\nslope = −{abs(b):.2f} ± 0.13 s.e.\n$P$ = 1.7 × 10$^{{-22}}$',transform=ax.transAxes,ha='right',va='top',fontsize=5.5)
    h=[Line2D([],[],marker='o',ls='-',lw=0.8,color=COL[s],markeredgecolor=COL[s],markersize=3,solid_capstyle='butt',label=nm(s)) for s in SP]
    leg=ax.legend(handles=h,loc='lower left',frameon=False,fontsize=5,handletextpad=0.2,labelspacing=0.2,handlelength=1.4)
    for t in leg.get_texts(): t.set_fontstyle('italic')
    lxc=df.lx-df.groupby('species').lx.transform('mean'); lyc=df.ly-df.groupby('species').ly.transform('mean')
    for s in SP:
        m=df.species==s; bx.scatter(lxc[m],lyc[m],s=10,color=COL[s],edgecolor=BG,linewidth=0.3,zorder=3)
    w,pw=F['w']; xx=np.linspace(lxc.min(),lxc.max(),10); bx.plot(xx,w*xx,color=FG,lw=0.8)
    bx.axhline(0,color=MUT,lw=0.4,zorder=0); bx.axvline(0,color=MUT,lw=0.4,zorder=0)
    bx.set_xlabel('log$_{10}$ chromosome size\n(centred on species mean)'); bx.set_ylabel('log$_{10}$ crossover rate\n(centred on species mean)')
    bx.text(0.97,0.97,f'Pooled within-species\nslope = −{abs(w):.2f} ± 0.26 s.e.\n$P$ = {pw:.1e}'.replace('e-0',' × 10$^{-').replace('$P$ = 2.7 × 10$^{-4','$P$ = 2.7 × 10$^{-4}$'),transform=bx.transAxes,ha='right',va='top',fontsize=5.5)
    for s in ('top','right'): bx.spines[s].set_visible(False); ax.spines[s].set_visible(False)
    fig.text(0.005,0.985,'a',fontsize=7,fontweight='bold',va='top'); fig.text(0.50,0.985,'b',fontsize=7,fontweight='bold',va='top')
    o='Suppl_Fig_MixedModel_CO_rate_vs_chrsize'; fig.savefig(o+'.svg',transparent=True); fig.savefig(o+'.pdf',facecolor=BG); fig.savefig(o+'.png',dpi=600,facecolor=BG); plt.close(fig)
fig3b('white'); fig3b('black'); suppl()
