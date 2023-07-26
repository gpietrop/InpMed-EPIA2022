from layer_integral import coastline
from commons.time_interval import TimeInterval
from instruments import bio_float
from basins import V2 as OGS
import matplotlib
# matplotlib.use("GTK3Agg")
import pylab as pl

clon, clat = coastline.get()
TI = TimeInterval('2015', "2016", "%Y")
ProfileList = bio_float.FloatSelector(None, TI, OGS.med)

nP = len(ProfileList)

Lon = [p.lon for p in ProfileList]
Lat = [p.lat for p in ProfileList]

fig, ax = pl.subplots()
fig.set_size_inches(10.0, 10.0 * 16 / 42)
ax.set_position([0.08, 0.13, 0.78, 0.78])

ax.plot(clon, clat, '-')

ax.plot(Lon, Lat, 'r.', markersize=4)
ax.set_xlim([-6, 36])
ax.set_ylim([30, 46])

fig.savefig('prova.png')
