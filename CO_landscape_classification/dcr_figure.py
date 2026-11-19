import pickle, statistics as st, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
rows,evr=pickle.load(open('dcr.pkl','rb'))
SP=['R. cephalotes','R. breviuscula','R. pubera','R. nervosa','R. ciliata','R. colorata','R. barbata']
COL={'distal-biased':'#eb6834','irregular':'#2a78d6'}; MK=dict(zip(SP,['o','s','h','^','D','v','P']))
plt.rcParams.update({'font.family':['Arial','Liberation Sans'],'font.size':6,'axes.linewidth':0.5,'xtick.major.width':0.5,'ytick.major.width':0.5,'xtick.major.size':2,'ytick.major.size':2,'pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(180/25.4,62/25.4),facecolor='white')
gs=fig.add_gridspec(1,3,width_ratios=[1.15,1,1],wspace=0.42,left=0.06,right=0.845,top=0.9,bottom=0.25)
# a
ax=fig.add_subplot(gs[0]); rng=np.random.default_rng(3)
DS={'R. cephalotes','R. breviuscula','R. pubera'}
D=[r['log2DCR'] for r in rows if r['species'] in DS]; I=[r['log2DCR'] for r in rows if r['species'] not in DS]
ax.axhspan(max(I),min(D),color='#bdbdbd',alpha=0.45,lw=0)
ax.axhline(0,color='#9a9a9a',lw=0.4,ls=(0,(2,2)))
for i,s in enumerate(SP):
    rr=[r for r in rows if r['species']==s]
    for r in rr: ax.plot(i+rng.uniform(-0.18,0.18),r['log2DCR'],MK[s],ms=3.6,mew=0.4,mec='white',color=COL[r['cluster']])
    m=st.mean(r['log2DCR'] for r in rr); ax.plot([i-0.3,i+0.3],[m,m],color='black',lw=0.9)
ax.set_xticks(range(7)); ax.set_xticklabels(SP,rotation=40,ha='right',fontstyle='italic'); ax.set_xlim(-0.6,6.6)
ax.set_ylabel('log$_2$ DCR (distal / central CO rate)')
for sname in ('top','right'): ax.spines[sname].set_visible(False)
ax.text(6.55,(max(I)+min(D))/2,'gap',fontsize=5,va='center',ha='right',color='#4d4d4d')
fig.text(0.005,0.95,'a',fontsize=8,fontweight='bold')
# b
ax=fig.add_subplot(gs[1]); x=[(k+0.5)*5 for k in range(10)]
for r in rows: ax.plot(x,r['profile'],color=COL[r['cluster']],lw=0.4,alpha=0.45)
for c in COL:
    prof=np.mean([r['profile'] for r in rows if r['cluster']==c],axis=0); ax.plot(x,prof,color=COL[c],lw=1.6)
ax.set_xlim(0,50); ax.set_xticks([0,10,20,30,40,50]); ax.set_xticklabels(['End','10','20','30','40','Centre'])
ax.set_xlabel('Position (% of chromosome length)'); ax.set_ylabel('CO rate (z-score within chromosome)'); ax.axhline(0,color='#9a9a9a',lw=0.4,ls=(0,(2,2)))
for sname in ('top','right'): ax.spines[sname].set_visible(False)
fig.text(0.355,0.95,'b',fontsize=8,fontweight='bold')
# c
ax=fig.add_subplot(gs[2])
for r in rows: ax.plot(r['PC1'],r['PC2'],MK[r['species']],ms=3.8,mew=0.4,mec='white',color=COL[r['cluster']])
ax.set_xlabel(f'PC1 ({100*evr[0]:.0f}% of variance)'); ax.set_ylabel(f'PC2 ({100*evr[1]:.0f}% of variance)')
for sname in ('top','right'): ax.spines[sname].set_visible(False)
fig.text(0.625,0.95,'c',fontsize=8,fontweight='bold')
h=[Line2D([],[],ls='none',marker='o',ms=4,color=COL[c],label=c.capitalize()) for c in COL]
h+=[Line2D([],[],ls='none',marker=MK[s],ms=3.8,mfc='white',mec='#4d4d4d',mew=0.6,label=s) for s in SP]
leg=fig.legend(handles=h,loc='center left',bbox_to_anchor=(0.86,0.57),frameon=False,fontsize=5.5,handletextpad=0.3,labelspacing=0.6,title='k-means (k = 2)',title_fontsize=5.5)
leg._legend_box.align='left'
for t in leg.get_texts()[2:]: t.set_fontstyle('italic')
out='Extended_Data_Fig_X_DCR'
fig.savefig(out+'.pdf'); fig.savefig(out+'.svg'); fig.savefig(out+'.png',dpi=300)
