import csv
from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon
D=''   # run in this folder after ceph_fission_ends.py
CL={1:46646626,2:39764246,3:35948395,4:32796034,5:30371766,6:30785530,7:30061336,8:28880904,9:28351248}
ORDER=[3,9,4,6,7,8,5,2,1]; WIN=0.2
C=defaultdict(list)
for i,l in enumerate(open(D+'R_cephalotes_CO_rates.txt')):
    if i: c,p,lo,r,hi=l.split(); C[int(c)].append((float(p),float(r)))
for c in C: C[c].sort()
rows=list(csv.DictReader(open(D+'ceph_end_classes.tsv'),delimiter='\t'))
cls={(int(r['chr']),r['end']):('fission-derived' if r['rearr_derived']=='True' else r['cls']) for r in rows}
F=[r for r in rows if r['rearr_derived']=='True']
pairs=[(a,next(b for b in rows if b['chr']==a['chr'] and b['end']!=a['end'])) for a in F]
pw=wilcoxon([float(a['mean_rate']) for a,b in pairs],[float(b['mean_rate']) for a,b in pairs],alternative='greater').pvalue
FG,MUT,LINE,AREA,ACC,CONS='#000000','#4d4d4d','#000000','#c8c8c8','#d4148f','#7a7a7a'
plt.rcParams.update({'font.family':['Arial','Liberation Sans'],'font.size':6,'axes.linewidth':0.5,'xtick.major.width':0.5,'ytick.major.width':0.5,'xtick.major.size':2,'ytick.major.size':2,'xtick.major.pad':1.5,'ytick.major.pad':1.5,'pdf.fonttype':42,'svg.fonttype':'none'})
W,H=180/25.4,100/25.4
fig=plt.figure(figsize=(W,H),facecolor='white')
gs=fig.add_gridspec(3,4,width_ratios=[1,1,1,0.95],wspace=0.32,hspace=0.62,left=0.065,right=0.985,top=0.86,bottom=0.1)
mk={'fission-derived':dict(marker='v',color=ACC,mfc=ACC),'conserved':dict(marker='v',color=CONS,mfc=CONS),'terminal rearrangement':dict(marker='v',color=CONS,mfc='none')}
for i,c in enumerate(ORDER):
    ax=fig.add_subplot(gs[i//3,i%3])
    L=CL[c]/1e6; xs=[p/1e6 for p,r in C[c]]; ys=[r for p,r in C[c]]
    for end in ('start','end'):
        if cls[(c,end)]=='fission-derived':
            a,b=(0,WIN*L) if end=='start' else ((1-WIN)*L,L); ax.axvspan(a,b,color=ACC,alpha=0.15,lw=0)
    if c==5:
        ax.axvline(3.55,color=FG,lw=0.5,ls=(0,(2,1.5)))
        ax.text(3.9,9.6,'Synteny junction',fontsize=5,va='top',ha='left')
    ax.fill_between(xs,ys,color=AREA,lw=0); ax.plot(xs,ys,color=LINE,lw=0.6)
    for end in ('start','end'):
        ax.plot([0 if end=='start' else L],[10.9],ms=4,mew=0.6,clip_on=False,ls='none',**mk[cls[(c,end)]])
    ax.set_xlim(0,L); ax.set_ylim(0,10); ax.set_yticks([0,5,10]); ax.set_xticks([t for t in (0,10,20,30,40) if t<=L])
    for s in ('top','right'): ax.spines[s].set_visible(False)
    ax.text(0.5,1.13,f'Chr{c}',transform=ax.transAxes,ha='center',va='bottom',fontsize=6)
    if i%3==0: ax.set_ylabel('CO rate (cM/Mb)')
    if i//3==2: ax.set_xlabel('Physical position (Mb)')
fig.text(0.005,0.965,'a',fontsize=8,fontweight='bold',va='center')
fig.text(0.025,0.965,'R. cephalotes',fontsize=7,fontstyle='italic',va='center')
handles=[Line2D([],[],ls='none',ms=4.5,mew=0.6,**mk[k],label=l) for k,l in (('fission-derived','Rearrangement-derived end'),('conserved','Conserved end'),('terminal rearrangement','End with terminal rearrangement'))]
fig.legend(handles=handles,loc='center left',bbox_to_anchor=(0.14,0.965),ncol=3,frameon=False,fontsize=6,handletextpad=0.3,columnspacing=1.4,borderaxespad=0)
ax=fig.add_subplot(gs[:,3])
for a,b in pairs:
    ya,yb=float(a['mean_rate']),float(b['mean_rate'])
    ax.plot([0,1],[yb,ya],color=MUT,lw=0.5)
    ax.plot([0],[yb],'o',ms=4,mew=0.6,color=CONS,mfc=CONS if b['cls']=='conserved' else 'white')
    ax.plot([1],[ya],'o',ms=4,mew=0.6,color=ACC,mfc=ACC)
    dy={'5':0.09,'6':-0.09}.get(a['chr'],0)
    ax.text(1.09,ya+dy,f"Chr{a['chr']}",va='center',fontsize=5.5,color=MUT)
ax.set_xlim(-0.35,1.45); ax.set_ylim(0,5.6); ax.set_xticks([0,1]); ax.set_xticklabels(['Opposite\nend','Rearrangement-\nderived end'])
ax.set_ylabel('Mean CO rate, terminal 20% of chromosome (cM/Mb)')
for s in ('top','right'): ax.spines[s].set_visible(False)
ax.text(-0.3,5.55,f'n = {len(pairs)} chromosomes\nWilcoxon signed-rank P = {pw:.4f}',ha='left',va='top',fontsize=6)
fig.text(0.735,0.965,'b',fontsize=8,fontweight='bold',va='center')
out='Extended_Data_Fig_X_Rcephalotes_fission_ends'
fig.savefig(out+'.pdf'); fig.savefig(out+'.svg'); fig.savefig(out+'.png',dpi=300)
