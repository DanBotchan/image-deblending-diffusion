from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
P=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(16,8))
fig.patch.set_facecolor('white'); ax.set(xlim=(0,160),ylim=(0,80)); ax.axis('off')
ink='#233345'; muted='#647386'; blue='#2876ad'; orange='#bc6831'; green='#288778'
def text(x,y,s,size=12,color=ink,weight='normal',ha='left'):
 ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va='center',linespacing=1.5)
def box(x,y,w,h,fc='#f5f7fa',ec='#dbe1e8'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.5,rounding_size=1',facecolor=fc,edgecolor=ec,lw=1))
def arrow(x,y,X,Y,color=muted,dashed=False):
 ax.annotate('',xy=(X,Y),xytext=(x,y),arrowprops=dict(arrowstyle='-|>',color=color,lw=1.5,linestyle='--' if dashed else '-'))
text(3,76,'One blend → two sources and a smooth mixing map',22,weight='bold')
text(3,71,'Training-free deblending with pretrained diffusion priors',12,muted)
text(3,65,'A   THE PIPELINE',11,muted,'bold')
for x,w in [(3,22),(31,36),(73,23),(102,28),(136,21)]:box(x,44,w,17)
text(5,57,'Input',14,weight='bold');text(5,51,'Blend y\nPrompts c₁, c₂',11)
text(33,57,'1  Separate semantics',14,blue,'bold');text(33,51,'Text-guided latent diffusion\n3 seeds → 3 candidate pairs',11)
text(75,57,'2  Select',14,weight='bold');text(75,51,'PickScore + prompts\nNo ground truth',10.5)
text(104,57,'3  Refine detail',14,green,'bold');text(104,51,'Pixel-space diffusion\nRefine selected pair',10.5)
text(138,57,'Recover',14,weight='bold');text(138,51,'Source 1 + source 2\nAlpha map',10.5)
for a,b in [(25,31),(67,73),(96,102),(130,136)]:arrow(a,52,b,52)
text(3,38,'B   INSIDE JOINT SAMPLING',11,muted,'bold')
text(157,38,'Shown for Stage 1; both stages use joint data consistency',10,muted,ha='right')
box(3,9,53,24,'#f6f9fc')
text(5,29,'Two branches · one shared pretrained prior',12,weight='bold')
box(5,12,23,12,'#eaf3fb','#b8d4e8');box(31,12,23,12,'#fff1e8','#e9cab6')
text(16.5,21,'Source 1',12,blue,'bold','center');text(16.5,16,'+ c₁   /   − c₂',12,blue,ha='center')
text(42.5,21,'Source 2',12,orange,'bold','center');text(42.5,16,'+ c₂   /   − c₁',12,orange,ha='center')
text(29.5,5,'Cross-negative prompts discourage source leakage',10,muted,ha='center')
arrow(56,21,65,21)
box(65,9,47,24,'#f4faf8','#c7dfd8')
text(88.5,29,'Decode estimates and reblend',12,weight='bold',ha='center')
text(88.5,22,r'$\hat{y}=\alpha_\phi\odot\hat{x}_1+(1-\alpha_\phi)\odot\hat{x}_2$',16,ha='center')
text(88.5,15,'Compare reconstructed blend with y',11,ha='center')
arrow(112,21,119,21)
box(119,9,38,24)
text(138,29,'Correct together',12,weight='bold',ha='center')
text(138,22,'Update both source samples',11,ha='center')
text(138,15,'Fit smooth α using 6 coefficients',10.5,green,ha='center')
# Return path is kept above the boxes, away from the forward flow.
ax.plot([138,138,29.5],[33.5,35,35],color=muted,lw=1.2,linestyle='--');arrow(29.5,35,29.5,33.5,dashed=True)
text(138,5,'Smooth α discourages edge leakage into the mask',9.5,muted,ha='center')
fig.subplots_adjust(0.01,0.01,0.99,0.99)
for ext in ['svg','pdf','png']:fig.savefig(P/f'method-proposal.{ext}',dpi=200,facecolor='white')
