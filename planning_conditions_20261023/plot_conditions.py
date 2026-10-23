"""Plot the joint conditions in the manuscript's restrained two-region palette."""
from pathlib import Path
import argparse
import os

HERE=Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR",str(HERE/".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


def main(output):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.3,"axes.labelsize":8.6,
                         "axes.linewidth":0.65,"xtick.labelsize":8,"ytick.labelsize":8,
                         "legend.fontsize":7.3,"savefig.facecolor":"white"})
    miso=pd.read_csv(HERE/"results/miso_joint_frontier.csv")
    ercot=pd.read_csv(HERE/"results/ercot_joint_frontier.csv")
    ms=pd.read_csv(HERE/"results/miso_binding_segments.csv")
    es=pd.read_csv(HERE/"results/ercot_binding_segments.csv")
    facts=__import__("json").loads((HERE/"results/ercot_scale_comparison.json").read_text())
    fig=plt.figure(figsize=(7.2,3.85))
    grid=fig.add_gridspec(2,2,left=.105,right=.975,top=.88,bottom=.25,
                          height_ratios=[10,0.85],hspace=.13,wspace=.38)
    colors=["#7A73B8","#E36C09"]
    for column,region in enumerate(["MISO","ERCOT"]):
        ax=fig.add_subplot(grid[0,column])
        strip=fig.add_subplot(grid[1,column],sharex=ax)
        color=colors[column]
        if region=="MISO":
            group=miso[miso.source_timing_rule.eq("linear")].sort_values("resource_share")
            bounds=miso.groupby("resource_share").all_year_growth_scale.agg(["min","max"])
            ax.fill_between(bounds.index*100,bounds["min"]*100,bounds["max"]*100,color=color,alpha=.16,lw=0)
            y=group.all_year_growth_scale.to_numpy()*100
            end=group.endpoint_growth_scale.to_numpy()*100
            segments=ms[ms.source_timing_rule.eq("linear")]
            ax.set(ylim=(0,205),yticks=[0,50,100,150,200],ylabel="Maximum reference growth (%)")
            ax.axhline(100,color="#D6D9DC",ls=":",lw=.7,zorder=0)
        else:
            group=ercot.sort_values("resource_share")
            y=group.all_year_provisional_fraction.to_numpy()*100
            end=group.endpoint_provisional_fraction.to_numpy()*100
            segments=es
            ax.set(ylim=(0,102),yticks=[0,25,50,75,100],ylabel="Required provisional response (%)")
        x=group.resource_share.to_numpy()*100
        ax.plot(x,y,color=color,lw=2.0,zorder=3)
        ax.plot(x,end,color="#595E62",ls=(0,(4,2.5)),lw=.9,zorder=4)
        positions=[0,250,500,750,1000]
        ax.scatter(x[positions],y[positions],color=color,s=15,edgecolors="white",linewidths=.55,zorder=5)
        ax.set(xlim=(0,100),xticks=[0,25,50,75,100])
        ax.grid(color="#E9E9E9",lw=.5,axis="both",zorder=0)
        ax.tick_params(axis="x",labelbottom=False,length=0)
        ax.spines[["top","right"]].set_visible(False)
        ax.text(-.17,1.11,"ab"[column],transform=ax.transAxes,fontsize=12,fontweight="bold")
        ax.set_title(region,loc="left",fontweight="bold",color=color,pad=10,fontsize=10)
        for s in segments.itertuples():
            left,right=100*s.resource_share_start,100*s.resource_share_end
            fill="#DAD4EB" if s.binding_year==2034 else ("#B8DBD2" if s.binding_year==2035 else "#EEEEEE")
            strip.axvspan(left,right,color=fill,lw=0)
            strip.text((left+right)/2,.5,str(s.binding_year) if s.binding_year else "None",
                       ha="center",va="center",fontsize=7,color="#30383B")
        strip.set(ylim=(0,1),yticks=[],xticks=[0,25,50,75,100])
        strip.tick_params(axis="x",length=2,width=.6,pad=3)
        strip.spines[["top","right","left"]].set_visible(False)
        strip.spines["bottom"].set_color("#999999")
        strip.text(-.03,.5,"Binding\nyear",transform=strip.transAxes,ha="right",va="center",fontsize=6.2)
    fig.supxlabel("Share of anticipated-to-prospective resource uplift (%)",x=.54,y=.105,fontsize=8.5)
    handles=[Line2D([0],[0],color="#333A3E",lw=1.8,label="All years"),
             Line2D([0],[0],color="#595E62",lw=.9,ls=(0,(4,2.5)),label="2035 only"),
             Patch(facecolor=colors[0],alpha=.16,edgecolor="none",label="MISO arrival-timing range")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.54,.003),ncol=3,
               frameon=False,handlelength=2.3,columnspacing=1.8)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=330)
    plt.close(fig)
    print(output)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"figures/joint_planning_conditions.png")
    main(parser.parse_args().output)
