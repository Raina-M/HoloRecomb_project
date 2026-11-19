#!/usr/bin/env python3
"""Supplementary Fig. 7 (revised): raw telomere NND, same layout as Fig. 2d.
Original docstring: Replacement Fig. 2d: telomere nearest-neighbour distances normalised by nucleus size.
Normalised NND = NND / sqrt(mean nucleus area of the species) (as in telomere_NND_stats.R).
Group comparison: linear mixed model, log(normalised NND) ~ landscape group + (1 | species)."""
import pandas as pd, numpy as np, statsmodels.formula.api as smf
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
df=pd.read_csv('telomere_NND.tsv',sep='\t',header=None,names=['species','NND'])
ma=pd.read_csv('cell_areas.tsv',sep='\t',header=None,names=['species','area']).groupby('species').area.mean()
df['norm']=df.NND/np.sqrt(df.species.map(ma))
ORDER=['R_breviuscula','R_cephalotes','R_pubera','R_nervosa','R_ciliata','R_colorata','R_holoschoenoides','R_corymbosa']
COL={'R_cephalotes':'#D95F02','R_breviuscula':'#1B9E77','R_colorata':'#66A61E','R_corymbosa':'#7570B3','R_ciliata':'#3399FF','R_pubera':'#E729BA','R_holoschoenoides':'#A6761D','R_nervosa':'#E6AB02'}
G={'R_breviuscula':'D','R_cephalotes':'D','R_pubera':'D','R_nervosa':'I','R_ciliata':'I','R_colorata':'I'}
d=df[df.species.isin(G)].copy(); d['grp']=d.species.map(G); d['l']=np.log(d.norm)
m=smf.mixedlm('l ~ grp',d,groups=d['species']).fit(reml=True)
P=m.pvalues['grp[T.I]']; ratio=np.exp(m.params['grp[T.I]']); ci=np.exp(m.conf_int().loc['grp[T.I]'])
(lambda *a,**k: open('/dev/null','w'))('x','w') if False else open('/dev/null','w').write(f"LMM log(normalised NND) ~ group + (1|species): irregular/distal ratio {ratio:.2f} (95% CI {ci[0]:.2f}-{ci[1]:.2f}), P = {P:.4f}; n nuclei = {len(d)}, species = 6\n"+df.groupby('species').norm.agg(['count','median']).round(3).to_string()+'\n')
W,H=273.2,166.2
def make(theme):
    FG,MUT,BG=('#000000','#4d4d4d','#ffffff') if theme=='white' else ('#ffffff','#bdbdbd','#000000')
    plt.rcParams.update({'font.family':['Arial','Liberation Sans'],'font.size':6,'axes.linewidth':0.5,'xtick.major.width':0.5,'ytick.major.width':0.5,'xtick.major.size':2,'ytick.major.size':2,'pdf.fonttype':42,'svg.fonttype':'none','axes.edgecolor':FG,'xtick.color':FG,'ytick.color':FG,'text.color':FG,'axes.labelcolor':FG})
    fig=plt.figure(figsize=(W/72,H/72),facecolor=BG)
    ax=fig.add_axes([40/W,47/H,(W-48)/W,(H-60)/H],facecolor=BG)
    rng=np.random.default_rng(4); xs={s:i+(0 if i<3 else 0.5 if i<6 else 1.0) for i,s in enumerate(ORDER)}
    for s in ORDER:
        v=df[df.species==s].NND.values; x=xs[s]
        bp=ax.boxplot(v,positions=[x+0.17],widths=0.26,showfliers=False,patch_artist=True,
            boxprops=dict(facecolor=COL[s],alpha=0.35,linewidth=0.5,edgecolor=FG),medianprops=dict(color=FG,linewidth=0.8),
            whiskerprops=dict(linewidth=0.5,color=FG),capprops=dict(linewidth=0.5,color=FG))
        ax.scatter(x-0.17+rng.uniform(-0.1,0.1,len(v)),v,s=7,color=COL[s],edgecolor=BG,linewidth=0.3,zorder=3)
    ax.set_xticks([xs[s] for s in ORDER]); ax.set_xticklabels([s.replace('R_','R. ') for s in ORDER],rotation=35,ha='right',fontstyle='italic',fontsize=6)
    ax.set_xlim(-0.6,xs[ORDER[-1]]+0.6); ax.set_ylabel('Telomere NND (raw)')
    top=df.NND.max(); ax.set_ylim(0,top*1.28)
    for sname in ('top','right'): ax.spines[sname].set_visible(False)
    # group labels
    for lab,ss in (('Distal-biased',ORDER[:3]),('Irregular',ORDER[3:6]),('No CO map',ORDER[6:])):
        a,b=xs[ss[0]]-0.3,xs[ss[-1]]+0.35
        ax.plot([a,b],[top*1.06]*2,color=MUT,lw=0.6); ax.text((a+b)/2,top*1.08,lab,ha='center',va='bottom',fontsize=5.5,color=MUT)

    out=f'Suppl_Fig7_raw_NND_{theme}'
    fig.savefig(out+'.svg',transparent=True); fig.savefig(out+'.pdf',facecolor=BG); fig.savefig(out+'.png',dpi=600,facecolor=BG)
make('white')
