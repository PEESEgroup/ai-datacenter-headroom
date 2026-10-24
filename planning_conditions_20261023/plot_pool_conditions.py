"""Plot current paired results and joint frontiers, not main Fig. 4c,d heatmaps."""
from pathlib import Path
import argparse
import os
os.environ.setdefault("MPLCONFIGDIR",str(Path(__file__).resolve().parent/".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import run_pool_consistency as p

HERE=Path(__file__).resolve().parent
PURPLE="#8075b5";ORANGE="#df720e";TEAL="#278b7c";GRAY="#6a6a6a"

def style(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(length=2.5,width=.6,pad=2)
    ax.grid(axis="y",color="#e6e3df",lw=.45,zorder=0)

def main(output):
    plt.rcParams.update({"font.family":"Arial","font.size":7.5,"axes.labelsize":7.5,
        "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":6.6,"axes.linewidth":.6,"pdf.fonttype":42})
    a=pd.read_csv(HERE/"pool_results/primary_annual.csv.gz")
    d=p.display(a)
    named=a.copy()
    for key,val in p.DISPLAY.items():
        if key!="data_center_scenario_id":named=named[named[key].eq(val)]
    low=pd.read_csv(HERE/"pool_results/lower_display_2035.csv")
    lower=pd.read_csv(HERE/"pool_results/lower_annual.csv.gz")
    for key,val in p.DISPLAY.items():
        if key!="data_center_scenario_id":lower=lower[lower[key].eq(val)]
    lower=lower[lower.source_DC_timing_rule.eq("linear")]
    fig=plt.figure(figsize=(7.15,5.85))
    for i,(region,color) in enumerate([("MISO",PURPLE),("ERCOT",ORANGE)]):
        x=.095+i*.505
        ax=fig.add_axes([x,.755,.39,.185]);style(ax)
        fig.text(.012+i*.505,.982,chr(97+i),fontsize=11,fontweight="bold",va="top")
        ax.set_title(region,loc="left",fontsize=8.5,pad=5)
        response="published_all" if region=="MISO" else "current_cap"
        g=a[a.planning_region.eq(region)&a.resource_condition.eq("anticipated")&a.response_condition.eq(response)]
        bounds=g.groupby("year").pressure_GW.agg(["min","max"])
        ax.fill_between(bounds.index,bounds["min"],bounds["max"],color=color,alpha=.15,lw=0)
        for name,ls in [("external_390_TWh_endpoint",(0,(4,2))),("external_430_TWh_endpoint",(0,(1,2)))]:
            g=lower[lower.planning_region.eq(region)&lower.resource_condition.eq("anticipated")&lower.response_condition.eq(response)&lower.data_center_scenario_id.eq(name)].sort_values("year")
            ax.plot(g.year,g.pressure_GW,color="#909090",ls=ls,lw=.8)
        for name,label,ls in [("lbnl_compounded_lower","Lower","--"),("lbnl_reference","Reference","-"),("lbnl_compounded_upper","Upper",":")]:
            g=named[named.planning_region.eq(region)&named.resource_condition.eq("anticipated")&named.response_condition.eq(response)&named.data_center_scenario_id.eq(name)].sort_values("year")
            ax.plot(g.year,g.pressure_GW,color=color,ls=ls,lw=1.1,marker="o" if name=="lbnl_reference" else None,ms=2.6,label=label)
            if name=="lbnl_reference":ax.annotate(f"{g.pressure_GW.iloc[-1]:.1f}",(2035,g.pressure_GW.iloc[-1]),xytext=(-2,5),textcoords="offset points",ha="right",fontsize=7.5,color=color,fontweight="bold")
        ax.set(xlim=(2025.6,2035.4),ylim=(0,40),yticks=[0,10,20,30,40],xticks=[2026,2030,2035],ylabel="DC pressure (GW)",xlabel="Year")
        ax.legend(ncol=3,frameon=False,loc="upper left",columnspacing=.65,handletextpad=.3,handlelength=1.5)
        bottom=fig.add_axes([x+.02,.565,.37,.10]);style(bottom)
        conditions=[("anticipated","published_all","Anticipated"),("prospective","published_all","Prospective")] if region=="MISO" else [("anticipated","current_cap","Current-only"),("anticipated","published_all","Published")]
        for y,(resource,rsp,label) in zip([1,0],conditions):
            g=a[a.planning_region.eq(region)&a.resource_condition.eq(resource)&a.response_condition.eq(rsp)&a.year.eq(2035)]
            r=d[d.planning_region.eq(region)&d.resource_condition.eq(resource)&d.response_condition.eq(rsp)&d.year.eq(2035)].iloc[0]
            for dy,field,marker,c in [(.17,"pressure_GW","D",GRAY),(-.17,"added_requirement_GW","o",color)]:
                lo,hi=g[field].min(),g[field].max()
                bottom.plot([lo,hi],[y+dy]*2,c=c,lw=1,alpha=.7)
                bottom.plot([lo,hi],[y+dy]*2,c=c,ls="none",marker="|",ms=4)
                bottom.scatter(r[field],y+dy,c=c,marker=marker,s=14,edgecolor="white",lw=.4,zorder=4)
        bottom.set(xlim=(-.6,40),xticks=[0,10,20,30,40],ylim=(-.5,1.5),yticks=[1,0],yticklabels=[r[2] for r in conditions],xlabel="2035 contribution (GW)")
        bottom.grid(False);bottom.grid(axis="x",color="#e6e3df",lw=.45);bottom.spines["left"].set_visible(False)
        bottom.tick_params(axis="y",length=0)
    fig.legend(handles=[Patch(facecolor="#aaa",alpha=.2,label="Core scenario range"),
        Line2D([],[],color="#909090",ls="--",label="390/430 TWh endpoints"),
        Line2D([],[],color=GRAY,marker="D",lw=1,ms=3,label="Pressure"),
        Line2D([],[],color=GRAY,marker="o",lw=1,ms=3,label="Added need")],loc="center",bbox_to_anchor=(.53,.485),ncol=4,frameon=False,columnspacing=1.3,handlelength=1.4)
    mc=pd.read_csv(HERE/"results/miso_joint_frontier.csv")
    ec=pd.read_csv(HERE/"results/ercot_joint_frontier.csv")
    native=pd.read_csv(HERE/"inputs/native_inputs.csv")
    for i,(region,color) in enumerate([("MISO",PURPLE),("ERCOT",ORANGE)]):
        x=.095+i*.505
        ax=fig.add_axes([x,.085,.39,.27]);style(ax)
        fig.text(.012+i*.505,.435,chr(99+i),fontsize=11,fontweight="bold",va="top")
        ax.set_title(region+(" (matched candidate)" if region=="MISO" else " (official demand)"),loc="left",fontsize=8.5,pad=6)
        g=native[native.planning_region.eq(region)&native.year.eq(2035)].iloc[0]
        uplift=(g.prospective_native_MW-g.anticipated_native_MW)/1000
        if region=="MISO":
            b=mc.groupby("resource_share").all_year_growth_scale.agg(["min","max"])
            lin=mc[mc.source_timing_rule.eq("linear")]
            ax.fill_between(b.index*100,b["min"]*100,b["max"]*100,color=color,alpha=.18,lw=0)
            ax.plot(lin.resource_share*100,lin.all_year_growth_scale*100,color=color,lw=1.5,label="All years")
            ax.plot(lin.resource_share*100,lin.endpoint_growth_scale*100,c=GRAY,ls="--",lw=1,label="2035 only")
            ax.axhline(100,c="#c8c8c8",lw=.6)
            ax.scatter(0,lin.all_year_growth_scale.iloc[0]*100,c=color,s=15,zorder=4)
            limit = lin.all_year_growth_scale.iloc[0] * 100
            ax.annotate(f"{limit:.1f}%",(0,limit),xytext=(6,-9),textcoords="offset points",color=color,fontsize=7)
            ax.set(ylabel="Candidate growth scale (%)",ylim=(0,210),yticks=[0,50,100,150,200])
            ax.legend(handles=[Line2D([],[],c=color,lw=1.5,label="All years, linear timing"),Line2D([],[],c=GRAY,ls="--",lw=1,label="2035 only"),Patch(facecolor=color,alpha=.18,label="Four timing rules")],frameon=False,loc="upper left",fontsize=6.5,handlelength=1.5)
        else:
            ax.plot(ec.resource_share*100,ec.all_year_provisional_fraction*100,color=color,lw=1.5)
            for u,value in [(0,87.6442),(.25,50.7189)]:
                j=(ec.resource_share-u).abs().idxmin();v=ec.loc[j,"all_year_provisional_fraction"]*100
                ax.scatter(u*100,v,c=color,s=17,zorder=4)
                ax.annotate(f"{v:.1f}%",(u*100,v),xytext=(5,4),textcoords="offset points",fontsize=7,color=color)
            ax.set(ylabel="Required provisional response (%)",ylim=(-2,103),yticks=[0,25,50,75,100])
            ax.legend(handles=[Line2D([],[],c=color,lw=1.5,label="Official demand: all years = 2035")],frameon=False,fontsize=6.3,loc="upper right",handlelength=1.3)
        ax.set(xlim=(-1,101),xticks=[0,25,50,75,100],xlabel="Share of native resource uplift (%)")
        sec=ax.secondary_xaxis("top",functions=(lambda z,delta=uplift:z*delta/100,lambda z,delta=uplift:z*100/delta))
        sec.set_xticks([u*uplift for u in [0,.25,.5,.75,1]])
        sec.set_xticklabels([f"{u*uplift:.1f}" for u in [0,.25,.5,.75,1]])
        sec.set_xlabel("2035 native resource uplift (GW)",fontsize=6.8,labelpad=3)
        sec.tick_params(labelsize=6.5,length=2,pad=2)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=420,facecolor="white",bbox_inches="tight",pad_inches=.04)
    plt.close(fig)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,default=HERE/"figures/Fig4_pool_consistent.png")
    main(ap.parse_args().output)
