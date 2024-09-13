import numpy as np
import matplotlib.pyplot as plt
import math


au2ev = 27.21139
wmin = 0/au2ev
wmax = 10/au2ev
ymax = 100
#freqs = np.linspace(wmin, wmax, 100)#[10:]
#_freqs = np.linspace(2/au2ev, 6/au2ev, 200)#[10:]
freqs = np.linspace(2/au2ev, 6/au2ev, 300)#[10:]
# freqs_coarse = np.concatenate((np.linspace(wmin, 10/au2ev, 4),np.linspace(10/au2ev,25/au2ev, 190),np.linspace(25/au2ev,wmax, 4)))

plt.figure(figsize = (7, 5))
fig, ax = plt.subplots()
# fig.suptitle(f"diamond TD-{mf.xc}/{szv}/{kmesh}", x=0.5)
ax.set_xlabel(r"$\omega$ (eV)", fontsize=11)
ax.set_xlim(wmin*au2ev, wmax*au2ev)
ax.set_ylabel(r"Im $\epsilon(\omega)$", fontsize=11)

factor = (8) * math.pi **2 /3

def lorentzian(x, mean, fwhm):
    eta = fwhm/2 #https://mathworld.wolfram.com/LorentzianFunction.html#:~:text=The%20Lorentzian%20function%20gives%20the,Lorentzian%20function%20has%20Fourier%20transform
    return 1.0/np.pi * eta / ((x-mean)**2 + eta**2)

fwhm = 0.15
lorentzian = 0 * factor * lorentzian(freqs*au2ev, 0, fwhm)

import json


# int_vel_sum = []
# with open("my_BSE_LDA_[3, 3, 3]_cc-pvdz_cvIvel_False_True.txt") as f:
#     for line in f:
#         if json.loads(line) < 1e-5:
#             int_vel_sum.append(float('nan'))
#         else:
#             int_vel_sum.append(json.loads(line))
# int_vel_sum = np.array(int_vel_sum)
# nan_mask = np.isfinite(int_vel_sum)
# ax.plot(freqs[nan_mask]*au2ev+0.10, (1/27)*factor*int_vel_sum[nan_mask]/((freqs[nan_mask]*au2ev)**2), '-', color='pink', label='$\Gamma$-centered 3x3x3')

int_vel_sum = []
with open("my_BSE_LDA_[5, 5, 5]_cc-pvdz_cvIvel_False_True.txt") as f:
    for line in f:
        if json.loads(line) < 1e-5:
            int_vel_sum.append(float('nan'))
        else:
            int_vel_sum.append(json.loads(line))
int_vel_sum = np.array(int_vel_sum)
nan_mask = np.isfinite(int_vel_sum)
ax.plot(freqs[nan_mask]*au2ev+0.05, (1/125)*factor*int_vel_sum[nan_mask]/((freqs[nan_mask]*au2ev)**2), '-', color='pink', label='$\Gamma$-centered       5x5x5')

int_vel_sum = []
with open("my_BSE_LDA_[7, 7, 7]_cc-pvdz_cvIvel_False_True.txt") as f:
    for line in f:
        if json.loads(line) < 1e-5:
            int_vel_sum.append(float('nan'))
        else:
            int_vel_sum.append(json.loads(line))
int_vel_sum = np.array(int_vel_sum)
nan_mask = np.isfinite(int_vel_sum)
ax.plot(freqs[nan_mask]*au2ev+0.02, (1/7**3)*factor*int_vel_sum[nan_mask]/((freqs[nan_mask]*au2ev)**2), '-', color='red', label='                        7x7x7')

# int_vel_sum = []
# with open("my_BSE_LDA_[3, 3, 3]_cc-pvdz_cvIvel_True_True.txt") as f:
#     for line in f:
#         if json.loads(line) < 1e-5:
#             int_vel_sum.append(float('nan'))
#         else:
#             int_vel_sum.append(json.loads(line))
# int_vel_sum = np.array(int_vel_sum)
# nan_mask = np.isfinite(int_vel_sum)
# ax.plot(freqs[nan_mask]*au2ev-0.46, (1/27)*factor*int_vel_sum[nan_mask]/((freqs[nan_mask]*au2ev)**2), '-', color='lightblue', label='shifted k-mesh 3x3x3')

int_vel_sum = []
with open("my_BSE_LDA_[5, 5, 5]_cc-pvdz_cvIvel_True_True.txt") as f:
    for line in f:
        if json.loads(line) < 1e-5:
            int_vel_sum.append(float('nan'))
        else:
            int_vel_sum.append(json.loads(line))
int_vel_sum = np.array(int_vel_sum)
nan_mask = np.isfinite(int_vel_sum)
ax.plot(freqs[nan_mask]*au2ev-0.11, (1/125)*factor*int_vel_sum[nan_mask]/((freqs[nan_mask]*au2ev)**2), linestyle=(0, (5, 1)), color='lightblue', label='shifted k-mesh 5x5x5')

int_vel_sum = []
with open("my_BSE_LDA_[7, 7, 7]_cc-pvdz_cvIvel_True_True.txt") as f:
    for line in f:
        if json.loads(line) < 1e-5:
            int_vel_sum.append(float('nan'))
        else:
            int_vel_sum.append(json.loads(line))
int_vel_sum = np.array(int_vel_sum)
nan_mask = np.isfinite(int_vel_sum)
ax.plot(freqs[nan_mask]*au2ev, (1/7**3)*factor*int_vel_sum[nan_mask]/((freqs[nan_mask]*au2ev)**2), linestyle=(0, (5, 1)), color='blue', label='                        7x7x7')

# ax.annotate(r'$\eta=0.15$ eV, $N_v=4$, $N_c=6$', [4.35, 380])
ax.set_ylim([0, ymax])
ax.set_xlim([2,6])
# ax.set_xlim([1.3,1.8])
ax.legend(ncol=1, fancybox=True, framealpha=0.9, loc='upper right', fontsize=9)
# plt.title('Si', fontsize=11)

fig.savefig("Si_BSE_555_777_LDA.pdf")
