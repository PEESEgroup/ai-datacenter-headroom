"""Plot the DC-only monthly evidence without modifying any main figure."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR",str(Path(__file__).resolve().parent/".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

ROOT=Path(__file__).resolve().parent
data=pd.read_csv(ROOT/"empirical_results/DC_monthly_vector_pairs.csv")
data=data[data.month.between("2025-01","2025-08")]
data["date"]=pd.to_datetime(data.month)
plt.rcParams.update({"font.family":"Arial","font.size":9,"axes.spines.top":False,
                     "axes.spines.right":False,"axes.labelsize":9,"axes.titlesize":10,
                     "xtick.labelsize":8,"ytick.labelsize":8,"savefig.facecolor":"white"})
figure,axes=plt.subplots(2,2,figsize=(7.2,4.3))
for letter,zone,axis in zip("abcd",["AEP","DOM","COMED","PS"],axes.flat):
    values=data[data.zone.eq(zone)]
    allowance=values.difference_allowance_MW.max()
    axis.axhspan(-allowance,allowance,color="#e8e8e8",zorder=0)
    axis.axhline(0,color="#555555",linewidth=.7,zorder=1)
    for vintage,color,style in [(2024,"#a85185","--"),(2025,"#238d82","-")]:
        rows=values[values.vintage.eq(vintage)].sort_values("date")
        if len(rows):
            axis.plot(rows.date,rows.actual_minus_forecast_MW,color=color,linestyle=style,
                      marker="o",markersize=3,linewidth=1.5,label=str(vintage)+" forecast")
    axis.set_title(zone,loc="left",fontweight="bold",pad=7)
    axis.text(-.055,1.05,letter,transform=axis.transAxes,fontweight="bold",fontsize=11,
              ha="right",va="bottom")
    axis.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,3,5,7]))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axis.grid(axis="y",color="#ededed",linewidth=.6)
    axis.set_xlabel("2025")
    axis.set_ylabel("Observed minus forecast (MW)")
handles,labels=axes[0,0].get_legend_handles_labels()
figure.legend(handles,labels,loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(.5,-.005))
figure.subplots_adjust(left=.11,right=.99,top=.91,bottom=.15,hspace=.62,wspace=.38)
(ROOT/"figures").mkdir(exist_ok=True)
figure.savefig(ROOT/"figures/PJM_DC_monthly_evidence.png",dpi=300)
plt.close(figure)
