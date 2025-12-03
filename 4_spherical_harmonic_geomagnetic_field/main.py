# LATITUDE: theta
# LONGITUDE: phi

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import time
from numpy import sin, cos, tan, pi, sqrt, atan2, arccos
from scipy.linalg import pinv
import os
from matplotlib import cm
import matplotlib as mpl
mpl.rcParams['savefig.directory'] = os.getcwd()

from mpl_toolkits.basemap import Basemap

# CONSTANTS
mu0 = 4*pi*1e-7
# radiusEarth = 6371.2e3  # mean radius of the Earth in meters
radiusEarth = 6.378137e6



# NOAA COEFFICIENTS
myOrderingCoeffs = [0,1,3,4,6,8,9,11,13,15,16,18,20,22,24,25,27,29,31,33,2,5,7,10,12,14,17,19,21,23,26,28,30,32,34]
df = pd.read_csv("noaa_coeffs.dat", sep='\s+', header=3)
print(df)
coeffs_1980 = []
for ele in myOrderingCoeffs:
    coeffs_1980.append(df.iloc[ele, 19]/1e9)
coeffs_1980 = np.array(coeffs_1980)

# READING DATA
df = pd.read_csv("field.dat", header=0, sep='\s+')

latitude_deg = df["latitude[deg]"].to_numpy()
longitude_deg = df["longitude[deg]"].to_numpy()
radialdist_km = df["radialdist[km]"].to_numpy()
bx_nT = df["bx[nT]"].to_numpy()
by_nT = df["by[nT]"].to_numpy()
bz_nT = df["bz[nT]"].to_numpy()

nMeasurements = latitude_deg.shape[0]


# convert to cartesian
longitude_rad = np.deg2rad(longitude_deg)
latitude_rad = np.deg2rad(latitude_deg)

# dataset defines latitude as 90 - \theta
# \theta = 90 - data point
latitude_rad = np.full(latitude_deg.shape, pi/2) - latitude_rad

radialdist_m = radialdist_km * 1e3

x_km = radialdist_km * np.sin(latitude_rad) * np.cos(longitude_rad)
y_km = radialdist_km * np.sin(latitude_rad) * np.sin(longitude_rad)
z_km = radialdist_km * np.cos(latitude_rad)

x_norm = x_km/1e3
y_norm = y_km/1e3
z_norm = z_km/1e3

# normalize field vectors
bx_norm = np.zeros((nMeasurements))
by_norm = np.zeros((nMeasurements))
bz_norm = np.zeros((nMeasurements))


bx_T = bx_nT / 1e9
by_T = by_nT / 1e9
bz_T = bz_nT / 1e9

for i in range(nMeasurements):
    _bx = bx_nT[i]
    _by = by_nT[i]
    _bz = bz_nT[i]
    mag = np.linalg.norm(np.array([_bx,_by,_bz]))
    bx_norm[i] = bx_nT[i] / mag
    by_norm[i] = by_nT[i] / mag
    bz_norm[i] = bz_nT[i] / mag

mag = sqrt(bx_norm**2 + by_norm**2 + bz_norm**2)
norm = Normalize(vmin=mag.min(), vmax=mag.max())
colors = cm.viridis(norm(mag))


fig = plt.subplot(projection="3d")
ax = fig.axes
plt.quiver(x_norm,y_norm,z_norm,bx_norm,by_norm,bz_norm, color=colors)
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
ax.set_zlim(-10,10)
ax.set_xlabel("X [$10^3$ km]")
ax.set_ylabel("Y [$10^3$ km]")
ax.set_zlabel("Z [$10^3$ km ]")
plt.show()

# sympy generated coefficent expressions
def sphHarmBx(r, theta, phi, norm_radius=False):
    if norm_radius:
        a = 1
    else:
        a = radiusEarth
    coeffs = np.array([
        +sin(theta),
    +cos(phi)*cos(theta),
    +sin(phi)*cos(theta),
    +(-3/2)*sin(2*theta),
    +sqrt(3)*cos(phi)*cos(2*theta),
    +sqrt(3)*sin(phi)*cos(2*theta),
    +(sqrt(3)/2)*cos(2*phi)*sin(2*phi),
    +(sqrt(3)/2)*sin(2*phi)*sin(2*theta),
    ])

    return coeffs.T

def sphHarmBy(r, theta, phi, norm_radius=False):
    if norm_radius:
        a = 1
    else:
        a = radiusEarth
    coeffs = np.array([
    np.zeros(theta.shape),
    +sin(phi),
    -cos(phi),
    np.zeros(theta.shape),
    +sqrt(3)*sin(phi)*cos(theta),
    -sqrt(3)*cos(phi)*cos(theta),
    +sqrt(3)*sin(2*phi)*sin(theta),
    -sqrt(3)*cos(phi)*sin(theta),
    ])

    return coeffs.T

def sphHarmBz(r, theta, phi, norm_radius=False):
    if norm_radius:
        a = 1
    else:
        a = radiusEarth

    coeffs = np.array([
    -2*cos(theta),
    -2*cos(phi)*sin(theta) ,
    -2*sin(phi)*sin(theta),
    -(3/2)*(3*cos(theta)**2-1),
    -((sqrt(3)*3)/2)*cos(phi)*sin(2*theta),
    -((sqrt(3)*3)/2)*sin(phi)*sin(2*theta),
    -((sqrt(3)*3)/2)*cos(2*phi)*sin(theta)**2,
    -((sqrt(3)*3)/2)*sin(2*phi)*sin(theta)**2,
    ])

    return coeffs.T



# coeffient matrix
x_coeffs = sphHarmBx(radialdist_m, latitude_rad, longitude_rad)
y_coeffs = sphHarmBy(radialdist_m, latitude_rad, longitude_rad)
z_coeffs = sphHarmBz(radialdist_m, latitude_rad, longitude_rad)

# only use certain coefficients
# x_coeffs = x_coeffs[:,:-10]
# y_coeffs = y_coeffs[:,:-10]
# z_coeffs = z_coeffs[:,:-10]

# stack them together to form the (3*NMeasurements)x15 matrix
A = np.vstack((x_coeffs, y_coeffs, z_coeffs))
# print(A.shape)
print(pd.DataFrame(A))
# print(A.max())

# magnetic field column vector
b = np.concat((bx_T, by_T, bz_T))
# print(pd.DataFrame(b))

# solve pseudo inverse
A_inverse = pinv(A)

solved_coeffs = A_inverse @ b

solved_coeffs_nT = solved_coeffs * 1e9
print("\n\nCoefficients [nT]:")
print(pd.DataFrame(solved_coeffs_nT))


def calcField(r,theta,phi,c):
    """Calculate the magnetic field vector at a point (r,theta,phi)

    Args:
        r (_type_): _description_
        theta (_type_): _description_
        phi (_type_): _description_
        c (_type_): the spherical harmonic coefficients

    Returns:
        field: (nPoints,3,15) (points, components, expansion coefficient)
    """

    # multiplying by coefficient vector `c` is element wise
    # field calc is (nPoints, nCoeffs)
    _c = c.reshape(1, c.shape[0])
    bx = sphHarmBx(r, theta, phi) * _c
    by = sphHarmBy(r, theta, phi) * _c
    bz = sphHarmBz(r, theta, phi) * _c
    _bxSum = np.sum(bx[0])
    field = np.array([bx,by,bz])
    field = np.sum(field, axis=2)
    return field.T

################################################################################################ 3D VECTOR PLOT
# # generate points on the surface of a sphere

# # # Number of points to generate
# # num_points = 600

# # # Use the golden spiral method for more uniform distribution
# # indices = np.arange(0, num_points) + 0.5
# # phi = (2 * np.pi / (1 + np.sqrt(5))) * indices  # Golden angle
# # theta = np.arccos(1 - 2 * indices / num_points)

# # lat = theta
# # lon = phi

# nPoints = 40 # has to be an even number
# lat = np.linspace(-90, 90, nPoints)
# lon = np.linspace(0, 360, nPoints)

# lon2d, lat2d = np.meshgrid(lon, lat)
# lat = lat2d.flatten()
# lon = lon2d.flatten()

# lat = np.deg2rad(lat)
# lon = np.deg2rad(lon)


# field_inter = calcField(radiusEarth, lat, lon, coeffs_1980)

# bx_plot = field_inter[:,0]
# by_plot = field_inter[:,1]
# bz_plot = field_inter[:,2]

# # norm_factor = 10
# # mag_plot = np.sqrt(bx_plot**2+by_plot**2+bz_plot**2)
# # bx_plot /= mag_plot*norm_factor
# # by_plot /= mag_plot*norm_factor
# # bz_plot /= mag_plot*norm_factor

# # convert to cartesian
# x_plot = radiusEarth * sin(lon) * cos(lat)
# y_plot = radiusEarth * sin(lon) * sin(lat)
# z_plot = radiusEarth * cos(lon)

# from matplotlib import cm
# magnitude = np.sqrt(bx_plot**2+by_plot**2+bz_plot**2)
# norm = Normalize(vmin=magnitude.min(), vmax=magnitude.max())
# colors = cm.viridis(norm(magnitude))

# x_plot /= radiusEarth
# y_plot /= radiusEarth
# z_plot /= radiusEarth

# # plot
# fig = plt.subplot(projection="3d")
# ax = fig.axes
# ax.quiver(x_plot, y_plot, z_plot, bx_plot, by_plot, bz_plot, color=colors, length=0.1, normalize=True)
# # ax.set_xlim(-10,10)
# # ax.set_ylim(-10,10)
# # ax.set_zlim(-10,10)
# ax.set_xlabel("X [$10^3$ km]")
# ax.set_ylabel("Y [$10^3$ km]")
# ax.set_zlabel("Z [$10^3$ km ]")
# plt.show()
################################################################################################ 3D VECTOR PLOT


################################################################################################ 2D CONTOUR PLOT
# nPoints = 200
# x1d = np.linspace(-8*radiusEarth, 8*radiusEarth , nPoints)
# y1d = np.linspace(radiusEarth, 8*radiusEarth, nPoints)

# x2d, y2d = np.meshgrid(x1d, y1d)

# x_inter = x2d.flatten()
# y_inter = y2d.flatten()
# z_inter = np.zeros(nPoints*nPoints)

# # only select certain points
# # mask = np.where(x_inter > 0.5*radiusEarth, True, False)
# # x_inter = np.nonzero(x_inter[mask])
# # y_inter = np.nonzero(y_inter[mask])
# # z_inter = np.nonzero(z_inter[mask])


# r_inter = sqrt(x_inter**2+y_inter**2+z_inter**2)
# theta_inter = arccos(z_inter/r_inter)
# phi_inter = atan2(y_inter, x_inter)

# field_inter = calcField(r_inter, theta_inter, phi_inter, coeffs_1980)

# b_north = field_inter[:,0]
# b_east = field_inter[:,1]
# b_downward = field_inter[:,2]

# b_north2d = b_north.reshape(nPoints, nPoints)
# b_east2d = b_east.reshape(nPoints, nPoints)
# b_downward2d = b_downward.reshape(nPoints, nPoints)

# bx = b_downward*sin(theta_inter)*cos(phi_inter) + b_east*cos(theta_inter)*cos(phi_inter)-b_north*np.sin(phi_inter)
# by = b_downward*sin(theta_inter)*sin(phi_inter) + b_east*cos(theta_inter)*sin(phi_inter)+b_north*cos(phi_inter)
# bz = b_downward*cos(theta_inter)-b_east*sin(theta_inter)

# bx2d = bx.reshape(nPoints, nPoints)
# by2d = by.reshape(nPoints, nPoints)
# bz2d = bz.reshape(nPoints, nPoints)

# magnitude2d = np.sqrt(bx2d**2+by2d**2+bz2d**2)

# PCM = plt.pcolormesh(x2d/radiusEarth, y2d/radiusEarth, magnitude2d*1e6, cmap="gist_rainbow")
# plt.colorbar(PCM, label=r"Magnitude [$\mu$T]")

# plt.streamplot(x2d/radiusEarth, y2d/radiusEarth, bx2d, by2d, color="black")

# plt.xlabel(r"X / $r_{Earth}$")
# plt.ylabel(r"Y / $r_{Earth}$")
# plt.show()


################################################################################################ 2D CONTOUR PLOT
# radii = np.linspace(1.1, 1.5, 10)
# radii = [1.0]

# for _r in radii:
#     nPoints = 360
#     map = Basemap(projection='moll', lon_0=0, lat_0=0, resolution='c')
#     # map = Basemap(projection='hammer', lon_0=0, resolution='c')
#     # map = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90, llcrnrlon=-180,urcrnrlon=180,resolution='c')



#     # Sample data points (latitude and longitude in degrees)
#     lat = np.linspace(-90, 90, nPoints)
#     lon = np.linspace(0, 180, nPoints)

#     lon2d, lat2d = np.meshgrid(lon, lat)
#     lat = lat2d.flatten()
#     lon = lon2d.flatten()


#     field_calc = calcField(_r*radiusEarth, np.deg2rad(lat), np.deg2rad(lon), solved_coeffs) ########### COEFFS

#     b_north = field_calc[:,0]
#     b_east = field_calc[:,1]
#     b_downward = field_calc[:,2]

#     b_north2d = b_north.reshape((nPoints,nPoints))
#     b_east2d = b_east.reshape((nPoints,nPoints))
#     b_downward2d = b_downward.reshape((nPoints,nPoints))

#     magnitude2d = sqrt(b_north2d**2+b_east2d**2+b_downward2d**2)

#     magnitudeLog2d = np.log10(magnitude2d)

#     # Plot the data points
#     map.contourf(lon2d, lat2d, magnitude2d*1e6, latlon=True, cmap="gist_rainbow")
#     map.drawcoastlines()
#     map.drawcountries()
#     # map.contourf(lon2d, lat2d, b_north2d*1e6, latlon=True, cmap="plasma")
#     # map.contourf(lon2d, lat2d, b_north2d, latlon=True)
#     plt.colorbar(label=r"Field [$\mu$T]")
#     plt.title('')
#     plt.show()

# showNorm = False
# if showNorm:
#     norm = Normalize(vmin=magnitude2d.min(), vmax=magnitude2d.max())
#     CF = plt.contourf(lon2d, lat2d, magnitude2d, cmap="gist_rainbow", norm=norm)
# else:
#     CF = plt.contourf(lon2d, lat2d, magnitude2d*1e6, cmap="gist_rainbow")

# plt.xlabel("Longitude (deg)")
# plt.ylabel("Latitude (deg)")
# plt.colorbar(CF, label=r'Field [$\mu$T]')
# plt.show()


nLat = 180
nLon = 360

lat = np.linspace(-90, 90, nLat)
lon = np.linspace(0, 360, nLon)

lon2d, lat2d = np.meshgrid(lon, lat)
lat = lat2d.flatten()
lon = lon2d.flatten()

# Golden spiral method for more uniform distribution
num_points = 1000
indices = np.arange(0, num_points) + 0.5
phi = np.arccos(1 - 2*indices/num_points)         # latitude (colatitude)
theta = np.pi * (1 + 5**0.5) * indices            # longitud


field_calc = calcField(radiusEarth, np.deg2rad(lat), np.deg2rad(lon), coeffs_1980) ########### COEFFS
# field_calc = calcField(radiusEarth, np.deg2rad(lat), np.deg2rad(lon), coeffs_1980) ########### COEFFS

b_north = field_calc[:,0]
b_east = field_calc[:,1]
b_downward = field_calc[:,2]

b_north2d = b_north.reshape((nLon,nLat))
b_east2d = b_east.reshape((nLon,nLat))
b_downward2d = b_downward.reshape((nLon,nLat))

magnitude2d = sqrt(b_north2d**2+b_east2d**2+b_downward2d**2)


# PCM = plt.contourf(lon2d, lat2d, magnitude2d*1e6, cmap="gist_rainbow")
# plt.colorbar(PCM)
# plt.xlabel("Longitude [deg]")
# plt.ylabel("Latitude [deg]")
# plt.show()







for i in range(coeffs_1980.shape[0]):
    c = coeffs_1980[i]
    r = solved_coeffs[i]
    ratio = r/c
    print(f"Relative Error {i+1}: {100*abs((ratio)-1):.2f} %")


# calculate dipole moment of the earth
dipoleMoment = (4*pi*radiusEarth**3 / mu0) * sqrt(solved_coeffs[0]**2 + solved_coeffs[1]**2+solved_coeffs[9]**2)
print(f"Dipole moment of earth: {float(dipoleMoment/1e22):.3f} x 10^22 Am^2")



theta = np.deg2rad(lat)
phi = np.deg2rad(lon)

# 1980
g10 = -29992e-9
g11 = -1956e-9
h11 = 5604e-9
g20 = -1997e-9
g21 = 3027e-9
g22 = 1663e-9
h21 = -2129e-9
h22 = 1663e-9

# # 2017
# g10 = -29424e-9
# g11 = -1475e-9
# h11 = 4736e-9
# g20 = -2466e-9
# g21 = 3002e-9
# g22 = 1679e-9
# h21 = -2901e-9
# h22 = 673e-9

# mine
# g10 = -29424e-9
# g11 = 1922e-9
# h11 = -5576e-9
# g20 = -1979e-9
# g21 = -1715e-9
# g22 = 483e-9
# h21 = 1237e-9
# h22 = -77e-9


bx = (
    +g10*sin(theta)
    +g11*cos(phi)*cos(theta)
    +h11*sin(phi)*cos(theta)
    +g20*(-3/2)*sin(2*theta)
    +g21*sqrt(3)*cos(phi)*cos(2*theta)
    +h21*sqrt(3)*sin(phi)*cos(2*theta)
    +g22*(sqrt(3)/2)*cos(2*phi)*sin(2*phi)
    +h22*(sqrt(3)/2)*sin(2*phi)*sin(2*theta)
)

by = (
    +g10*0
    +g11*sin(phi)
    -h11*cos(phi)
    +g22*0
    +g21*sqrt(3)*sin(phi)*cos(theta)
    -h21*sqrt(3)*cos(phi)*cos(theta)
    +g22*sqrt(3)*sin(2*phi)*sin(theta)
    -h22*sqrt(3)*cos(phi)*sin(theta)
)

bz = (
    -g10*2*cos(theta)
    -g11*2*cos(phi)*sin(theta)
    -h11*2*sin(phi)*sin(theta)
    -g20*(3/2)*(3*cos(theta)**2-1)
    -g21*((sqrt(3)*3)/2)*cos(phi)*sin(2*theta)
    -h21*((sqrt(3)*3)/2)*sin(phi)*sin(2*theta)
    -g22*((sqrt(3)*3)/2)*cos(2*phi)*sin(theta)**2
    -h22*((sqrt(3)*3)/2)*sin(2*phi)*sin(theta)**2
)

magnitude = sqrt(bx**2+by**2+bz**2)
magnitude2d = magnitude.reshape((nLat,nLon))

levels = np.arange(40, 70, 0.5)
# levels = 20




map = Basemap(projection='moll', lon_0=0, lat_0=0, resolution='c')
PCM = map.contourf(lon2d, lat2d, magnitude2d*1e6, cmap="gist_rainbow", latlon=True, levels=levels)
map.drawcoastlines()
pcm = map.contour(lon2d, lat2d, magnitude2d*1e6, ls="dashed", latlon=True, levels=levels, colors="black")
# plt.clabel(PCM, inline=True, colors="black", fontsize=6, fmt="%.2f")

# PCM = plt.contourf(lon2d, lat2d, magnitude2d*1e6, cmap="gist_rainbow")
plt.colorbar(PCM, label=r"Magnitude [$\mu$T]")
# plt.xlabel("Longitude [deg]")
# plt.ylabel("Latitude [deg]")
plt.show()
