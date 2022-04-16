import sys
sys.path.append("..")
from uqgrid.psysdef import Psystem
from uqgrid.parse import load_matpower, load_psse, load_gic, add_dyr
from uqgrid.pflow import runpf
import matplotlib.pyplot as plt
import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import networkx as nx

from shapely.geometry import Point

#psys = load_psse(raw_filename="../data/ACTIVSg200.raw")
psys = load_psse(raw_filename="../data/ACTIVSg2000.raw")
add_dyr(psys, "../data/ACTIVSg2000.dyr")
#sub = load_gic(psys, "../data/ACTIVSg200.gic")
sub = load_gic(psys, "../data/ACTIVSg2000.gic")
#psys.plot_network()
#plt.show()


df = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))
df = df[df["state"] == "Texas"]
ax = gplt.polyplot(df, edgecolor='None', facecolor='lightgray')
pos = {i:psys.substations[psys.bus2sub[i]] for i in range(psys.nbuses)}

#nx.draw_networkx_nodes(psys.graph, pos=pos, ax=ax, node_size=40, node_color="black")
#nx.draw_networkx_edges(psys.graph, pos=pos, ax=ax, alpha=0.5, width=1)

gen_buses = [gen.bus for gen in psys.gens]
#nx.draw_networkx_nodes(psys.graph, pos=pos, node_size=50, nodelist=gen_buses, node_color="tab:red")


#plabels = [str(gen.bus) for gen in psys.gens]
#ppts = [Point(pos[i][0], pos[i][1]) for i in gen_buses]
#dgens = {'col1': plabels, 'geometry': ppts}
#gdf = gpd.GeoDataFrame(dgens, crs="EPSG:4326")

#gplt.kdeplot(
#    gdf,
#    cmap='Reds',
#    shade=True, thresh=0.05,
#    clip=df.geometry,
#    ax=ax
#)
#plt.savefig("texas3.pdf")

genrou_buses = [gen.bus for gen in psys.gendyn]
genrou_labels = [str(gen.bus) for gen in psys.gendyn]
genrou_H = [gen.H for gen in psys.gendyn]
ppts = [Point(pos[i][0], pos[i][1]) for i in genrou_buses]
dgens = {'col1': genrou_labels, 'geometry': ppts, 'inertia':genrou_H}
gdf = gpd.GeoDataFrame(dgens, crs="EPSG:4326")

gplt.pointplot(
    gdf,
    hue='inertia', scale ='inertia', cmap='inferno_r',
    legend=True,
    limits = (10, 30),
    ax=ax
)
plt.savefig("Texas4.pdf")

plt.show()
