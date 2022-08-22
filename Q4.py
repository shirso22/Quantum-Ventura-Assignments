"""Polarization dependent mirror with rectangular hole design"""
import grcwa
import numpy as np

# Truncation order (actual number might be smaller)
nG = 301
# lattice constants
L1 = [1.5,0]
L2 = [0,1.5]
# frequency and angles
freq = 0.527/1.5
theta = 0.
phi = 0.
# to avoid singular matrix, alternatively, one can add fictitious small loss to vacuum
Qabs = np.inf
freqcmp = freq*(1+1j/2/Qabs)
# the patterned layer has a griding: Nx*Ny
Nx = 400
Ny = 400

# now consider 3 layers: vacuum + patterned + vacuum
ep0 = 1. # dielectric for layer 1 (uniform)
epp = 12. # dielectric for patterned layer
epbkg = 1. # dielectric for holes in the patterned layer 
epN = 1.  # dielectric for layer N (uniform)

thick0 = 1. # thickness for vacuum layer 1
thickp = 0.55 # thickness of patterned layer
thickN = 1.

# patterned layer with rectangular holes 
ax=0.4
ay=0.12
epgrid2 = np.ones((Nx,Ny))*epp
x0 = np.linspace(0,1.,Nx)
y0 = np.linspace(0,1.,Ny)
x, y = np.meshgrid(x0,y0,indexing='ij')
ind = np.logical_and(np.abs(x-.5)<ax/2, np.abs(y-.5)<ay/2)
epgrid2[ind]=epbkg

######### setting up RCWA
obj = grcwa.obj(nG,L1,L2,freqcmp,theta,phi,verbose=1)
# input layer information
obj.Add_LayerUniform(thick0,ep0)
obj.Add_LayerGrid(thickp,Nx,Ny)
obj.Add_LayerUniform(thickN,epN)
obj.Init_Setup()

# planewave excitation
planewave={'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}
obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)
# eps in patterned layer
obj.GridLayer_geteps(epgrid2.flatten())

# compute reflection and transmission
R,T= obj.RT_Solve(normalize=1)
print('R=',R,', T=',T,', R+T=',R+T)

# compute reflection and transmission by order
# Ri(Ti) has length obj.nG, too see which order, check obj.G; too see which kx,ky, check obj.kx obj.ky
Ri,Ti= obj.RT_Solve(normalize=1,byorder=1)
