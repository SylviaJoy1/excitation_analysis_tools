import numpy as np
import sys
import matplotlib.pyplot as plt

formula = 'Si'

sys_args =  sys.argv[1:]
#fnl = str(sys_args[2])
fnl = 'LDA'
random_shift = str(sys_args[0])
if random_shift == 'True':
    random_shift = True
else:
    random_shift = False
#if random_shift is None:
#    random_shift = False
basis = str(sys_args[1])
#if basis is None:
#    basis = 'gth-dzvp'
kdensity = int(sys_args[2])
dipole_corr = str(sys_args[3])

kmesh = [kdensity, kdensity, kdensity]

au2ev = 27.21139
wmin = 2/au2ev
wmax = 6/au2ev
freqs = np.linspace(wmin, wmax, 200)
#freqs_coarse = np.concatenate((np.linspace(wmin, 10/au2ev, 150),np.linspace(10/au2ev,25/au2ev, 250),np.linspace(25/au2ev,wmax, 10)))
freqs_coarse = np.linspace(wmin,wmax, 300)

intensities_vel_cv0 = []
intensities_vel_cv1 = []
intensities_vel_cv2 = []
int_vel_cv_sum = []
for w in range(1,len(freqs_coarse)+1):
    intensities_vel_cv0_w = f'my_BSE_{fnl}_{kmesh}_{basis}_cvIvel0_{random_shift}_{dipole_corr}_{w}.txt'
    try:
        intensities_vel_cv0.append(np.loadtxt(intensities_vel_cv0_w))
    except:
        intensities_vel_cv0.append(0)
    intensities_vel_cv1_w = f'my_BSE_{fnl}_{kmesh}_{basis}_cvIvel1_{random_shift}_{dipole_corr}_{w}.txt'
    try:
        intensities_vel_cv1.append(np.loadtxt(intensities_vel_cv1_w))
    except:
        intensities_vel_cv1.append(0)
    intensities_vel_cv2_w = f'my_BSE_{fnl}_{kmesh}_{basis}_cvIvel2_{random_shift}_{dipole_corr}_{w}.txt'
    try:
        intensities_vel_cv2.append(np.loadtxt(intensities_vel_cv2_w))
    except:
        intensities_vel_cv2.append(0)
    intensities_vel_cv_sum_w = f'my_BSE_{fnl}_{kmesh}_{basis}_cvIvel_{random_shift}_{dipole_corr}_{w}.txt'
    try:
        int_vel_cv_sum.append(np.loadtxt(intensities_vel_cv_sum_w))
    except:
        int_vel_cv_sum.append(0)

fig, ax = plt.subplots(1,1, layout="constrained", figsize=(4,6))
fig.suptitle(f"Si BSE@GW@{fnl}/{basis}/{kdensity}x{kdensity}x{kdensity}, shift={random_shift}", x=0.5)
ax.set_xlabel(r"Energy (eV)")
ax.set_xlim(wmin*au2ev, wmax*au2ev)
ax.set_ylabel(r"Polarizability $\alpha(E)$ (atomic units)")
#ax.set_ylim(ymin, ymax)

cv = True
if cv:
    np.savetxt('my_BSE_{}_{}_{}_cvIvel0_{}_{}.txt'.format(fnl, kmesh, basis, random_shift, dipole_corr), intensities_vel_cv0)
    ax.scatter(freqs_coarse*au2ev, intensities_vel_cv0/(freqs_coarse*au2ev)**1, facecolor='none', edgecolor='C0', label='CV xx')
    ax.plot(freqs_coarse*au2ev, intensities_vel_cv0/(freqs_coarse*au2ev)**1, color='C0')
    np.savetxt('my_BSE_{}_{}_{}_cvIvel1_{}_{}.txt'.format(fnl, kmesh, basis, random_shift, dipole_corr), intensities_vel_cv1)
    ax.scatter(freqs_coarse*au2ev, intensities_vel_cv1/(freqs_coarse*au2ev)**1, facecolor='none', edgecolor='C1', label='CV yy')
    ax.plot(freqs_coarse*au2ev, intensities_vel_cv1/(freqs_coarse*au2ev)**1, color='C1')
    np.savetxt('my_BSE_{}_{}_{}_cvIvel2_{}_{}.txt'.format(fnl, kmesh, basis, random_shift, dipole_corr), intensities_vel_cv2)
    ax.scatter(freqs_coarse*au2ev, intensities_vel_cv2/(freqs_coarse*au2ev)**1, facecolor='none', edgecolor='C2', label='CV zz')
    ax.plot(freqs_coarse*au2ev, intensities_vel_cv2/(freqs_coarse*au2ev)**1, color='C2')
    np.savetxt('my_BSE_{}_{}_{}_cvIvel_{}_{}.txt'.format(fnl, kmesh, basis, random_shift, dipole_corr), int_vel_cv_sum)
    ax.scatter(freqs_coarse*au2ev, int_vel_cv_sum/(freqs_coarse*au2ev)**1, facecolor='none', edgecolor='black', label='CV sum')
    ax.plot(freqs_coarse*au2ev, int_vel_cv_sum/(freqs_coarse*au2ev)**1, color='black')
ax.legend()

if random_shift:
    fig.savefig(f"{dipole_corr}_{fnl}_{formula}_my_BSE_{basis}_{kmesh}_random_shift.pdf")
else:
    fig.savefig(f"{dipole_corr}_{fnl}_{formula}_my_BSE_{basis}_{kmesh}.pdf")
