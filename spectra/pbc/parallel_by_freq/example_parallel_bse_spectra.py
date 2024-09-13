import numpy as np
import sys
import matplotlib.pyplot as plt
from pyscf.pbc import gto, dft, scf, df
from pyscf.pbc.gw import krgw_ac
sys.path.append('/burg/berkelbach/users/sjb2225/v2.4.0/bse_master/static_BSE/periodic')
sys.path.append('/burg/berkelbach/users/sjb2225/v2.4.0/bse_master/pyscf-spectra')
from kbse import BSE
import parallel_pbc_spectra as pbc_spectra
from pyscf.pbc.scf import chkfile

#WARNING!
#PySCF TDA gives eigenvector that is complex conj of what it should be for spectra
#Until this is fixed, you will need to manually take the complex conj of the eigenvector
#My BSE code gives the correction conjugation

import sys
sys_args =  sys.argv[1:]
#fnl = str(sys_args[2])
fnl = 'LDA'
w = int(sys_args[0])
random_shift = str(sys_args[1])
if random_shift == 'True':
    random_shift = True
else:
    random_shift = False
basis = str(sys_args[2])
#    basis = 'gth-dzvp'
kdensity = int(sys_args[3])
dipole_corr = str(sys_args[4])
print('fnl, random_shift, basis, kdensity, dipole_corr', fnl, random_shift, basis, kdensity, dipole_corr)
from pyscf.pbc.tools import pyscf_ase, lattice
from pyscf.pbc.tools.pbc import super_cell
formula = 'Si'
ase_atom = lattice.get_ase_atom(formula)
cell = gto.Cell()
cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
cell.a = ase_atom.cell
cell.unit = 'B'
cell.basis = basis
#cell.pseudo = 'gth-pade'
#cell.ke_cutoff = 150
cell.build(verbose=9)

kmesh = [kdensity, kdensity, kdensity]
scaled_center = [0.0, 0.0, 0.0]
if random_shift:
    scaled_center = [0.11, 0.21, 0.31]
kpts = cell.make_kpts(kmesh, scaled_center=scaled_center)

au2ev = 27.21139
wmin = 2/au2ev
wmax = 6/au2ev
fwhm = 0.15/au2ev
freqs_coarse = np.linspace(wmin,wmax, 300)
freqs_coarse = np.array([freqs_coarse[w-1]])
#PBC kpt sampling my CIS
#################
import os
gdf = df.GDF(cell, kpts)
gdf_fname = 'parallel_{}_shift_{}_{}_{}.h5'.format(random_shift, fnl, kmesh, basis)
gdf._cderi_to_save = gdf_fname
if not os.path.isfile(gdf_fname):
    gdf.build()

chkfname = 'parallel_GDF_{}_shift_{}_{}_{}.chk'.format(random_shift, fnl, kmesh, basis)
if os.path.isfile(chkfname):
    #mf = scf.KRHF(cell, kpts, exxdiv=None)
    mf = dft.KRKS(cell, kpts, exxdiv=None).density_fit()
    #mf.with_df = gdf
    #mf.with_df._cderi = gdf_fname
    mf.xc = fnl
    data = chkfile.load(chkfname, 'scf')
    mf.__dict__.update(data)
else:
    #mf = scf.KRHF(cell, kpts, exxdiv=None)
    mf = dft.KRKS(cell, kpts, exxdiv=None).density_fit()
    #mf.with_df = gdf
    #mf.with_df._cderi = gdf_fname
    mf.xc = fnl
    mf.conv_tol = 1e-12
    mf.chkfile = chkfname
    mf.verbose = 9
    mf.kernel()
from pyscf.lib import chkfile
mygw = krgw_ac.KRGWAC(mf)
#INPUT OPTIONS for GW
nocc = cell.nelectron//2
lowest_gw_occ = max(0,nocc - 4)
highest_gw_vir = nocc + 6
orbs = range(lowest_gw_occ, highest_gw_vir)
gw_chkfile = f'gw_{fnl}_{kmesh}_{basis}_{random_shift}.chk'
if os.path.isfile(gw_chkfile):
    mygw.mo_energy =  chkfile.load(gw_chkfile, 'mo_energy')
else:
    mygw.linearized = True
    mygw.ac = 'pade'
    mygw.fc = False
    nocc = mygw.nocc
    mygw.kernel(orbs=orbs, nw=80)
    chkfile.dump(gw_chkfile, 'mo_energy', mygw.mo_energy)

mytd = BSE(mygw, TDA=True, singlet=True, CIS=False)
intensities_vel_cv = pbc_spectra.polarizability_corrvec_tda(mytd, freqs_coarse, fwhm=fwhm, BSE=True, orbs=orbs, imds_chkfile=f'imds_{fnl}_{kmesh}_{basis}_{random_shift}.chk')

#from pyscf.pbc import ci 
#mytd = ci.KCIS(mf)
#intensities_vel_cv = pbc_spectra.polarizability_corrvec_tda(mytd, freqs_coarse, fwhm=fwhm, BSE=False, imds_chkfile=f'{fnl}_{kmesh}_{basis}_{random_shift}.chk')

int_vel_cv_sum = np.sum(intensities_vel_cv, axis=0)

#save cv spectra at freq w
np.savetxt(f'my_BSE_{fnl}_{kmesh}_{basis}_cvIvel0_{random_shift}_{dipole_corr}_{w}.txt', intensities_vel_cv[0])
np.savetxt(f'my_BSE_{fnl}_{kmesh}_{basis}_cvIvel1_{random_shift}_{dipole_corr}_{w}.txt', intensities_vel_cv[1])
np.savetxt(f'my_BSE_{fnl}_{kmesh}_{basis}_cvIvel2_{random_shift}_{dipole_corr}_{w}.txt', intensities_vel_cv[2])
np.savetxt(f'my_BSE_{fnl}_{kmesh}_{basis}_cvIvel_{random_shift}_{dipole_corr}_{w}.txt', int_vel_cv_sum)
