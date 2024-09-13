import numpy as np
from pyscf import gto, scf, dft, tddft

mol = gto.Mole()
mol.build(
    atom = '''
        O        0.000000    0.000000    0.117790
        H        0.000000    0.755453   -0.471161
        H        0.000000   -0.755453   -0.471161''',
    basis = '6-31G',
    symmetry = True,
)

mf = dft.RKS(mol)
# mf.xc = 'PBE0'
mf.xc = 'HF'
mf.kernel()

#mytd = tddft.TDDFT(mf)
mytd = tddft.TDA(mf)
mytd.nstates = 10
mytd.kernel()
mytd.verbose = 7
mytd.analyze()

import spectra
au2ev = 27.21139
wmin = 0/au2ev
wmax = 25/au2ev
freqs = np.linspace(wmin, wmax, 200)
intensities = spectra.polarizability_tda(mytd, freqs, fwhm=1.0/au2ev)
int_sum = np.sum(intensities, axis=0)
intensities_vel = spectra.polarizability_tda(mytd, freqs, fwhm=1.0/au2ev, gauge='velocity')
int_vel_sum = np.sum(intensities_vel, axis=0)

freqs_coarse = freqs[::10]
intensities_cv = spectra.polarizability_corrvec_tda(mytd, freqs_coarse, fwhm=1.0/au2ev)
int_cv_sum = np.sum(intensities_cv, axis=0)
intensities_vel_cv = spectra.polarizability_corrvec_tda(mytd, freqs_coarse, fwhm=1.0/au2ev, gauge='velocity')
int_vel_cv_sum = np.sum(intensities_vel_cv, axis=0)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2,1, layout="constrained", figsize=(4,6))

fig.suptitle(f"H$_2$O TD-{mf.xc}/{mol.basis}", x=0.5)

for i in range(2):
    ax[i].set_xlabel(r"Energy (eV)")
    ax[i].set_xlim(wmin*au2ev, wmax*au2ev)
    ax[i].set_ylabel(r"Polarizability $\alpha(E)$ (atomic units)")
    ax[i].set_ylim(-1,25)

ax[0].set_title('Length gauge')
ax[0].plot(freqs*au2ev, intensities[0], '-', color='C0', label='xx')
ax[0].plot(freqs*au2ev, intensities[1], '-', color='C1', label='yy')
ax[0].plot(freqs*au2ev, intensities[2], '-', color='C2', label='zz')
ax[0].plot(freqs*au2ev, int_sum, '-', color='black', label='sum')
ax[0].plot(freqs_coarse*au2ev, intensities_cv[0], 'o', color='C0', label='CV xx')
ax[0].plot(freqs_coarse*au2ev, intensities_cv[1], 'o', color='C1', label='CV yy')
ax[0].plot(freqs_coarse*au2ev, intensities_cv[2], 'o', color='C2', label='CV zz')
ax[0].plot(freqs_coarse*au2ev, int_cv_sum, 'o', color='black', label='CV sum')
ax[0].legend()

ax[1].set_title('Velocity gauge')
ax[1].plot(freqs*au2ev, intensities_vel[0], '-', color='C0', label='xx')
ax[1].plot(freqs*au2ev, intensities_vel[1], '-', color='C1', label='yy')
ax[1].plot(freqs*au2ev, intensities_vel[2], '-', color='C2', label='zz')
ax[1].plot(freqs*au2ev, int_vel_sum, '-', color='black', label='sum')
ax[1].plot(freqs_coarse*au2ev, intensities_vel_cv[0], 'o', color='C0', label='CV xx')
ax[1].plot(freqs_coarse*au2ev, intensities_vel_cv[1], 'o', color='C1', label='CV yy')
ax[1].plot(freqs_coarse*au2ev, intensities_vel_cv[2], 'o', color='C2', label='CV zz')
ax[1].plot(freqs_coarse*au2ev, int_vel_cv_sum, 'o', color='black', label='CV sum')
ax[1].legend()

# fig.savefig("plot_h2o-pol_pbe0.pdf")
fig.savefig("plot_h2o-pol_hf.pdf")
