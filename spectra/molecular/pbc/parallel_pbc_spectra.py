import numpy as np
import scipy
import time
from pyscf import lib
import os
from pyscf.lib import chkfile

FWHM = 0.2/27.21139

import sys
sys.path.append('/burg/berkelbach/users/sjb2225/v2.4.0/bse_master/static_BSE/periodic')
import kbse

def spectrum_corrvec(hop, ops, freqs, hdiag=None, fwhm=FWHM):
    """Calculate a generic spectrum using the correction vector method.

    Args: 
        hop: Function for hamiltonian matrix-vector multiplication.  
        ops: Operators defining the response function. Can be a list of operators.
        freqs: Array of frequencies where the spectrum will be calculated.  
        hdiag: Diagonal of the hamiltonian matrix, to be used in preconditioning.

    Returns:
        Spectral intensity at each frequency. Only the diagonal elements of the
        operator response are calculated.
    """

    eta = fwhm/2

    if len(np.array(ops).shape) == 2:
        nops, size = np.array(ops).shape
    else:
        nops, size = 1, len(ops)
        ops = list(ops)
    
    intensities = np.zeros((nops,len(freqs)))

    for w, freq in enumerate(freqs):
        def matvec(x):
            hx = hop([x])
            return (freq**2 + eta**2)*x - 2*freq*hx + hop(hx)
        t0 = time.process_time()
        #TODO: implement TDDFT instead of just TDA
        A = scipy.sparse.linalg.LinearOperator((size,size), matvec=matvec)

        M = None
        if hdiag is not None:
            def precond(x):#x/(-diag+omega+ieta)
                return x/(freq**2 - 2*freq*hdiag + hdiag*hdiag + eta**2)
            M = scipy.sparse.linalg.LinearOperator((size,size), matvec=precond)

        for i in range(nops):
            x0 = precond(ops[i].conj())
            xw_i, exit_code = scipy.sparse.linalg.gcrotmk(A, ops[i].conj(), x0=x0, M=M, tol=1e-2)#0.01 is good enough
            #xw_i, exit_code = scipy.sparse.linalg.gmres(A, ops[i], x0=x0, maxiter=50, M=M, callback_type = 'pr_norm')

            if exit_code != 0:
                print(f"Warning! GCROT(m,k) not converged after {exit_code} iterations")
            intensities[i,w] = 1/np.pi * eta * np.dot(ops[i], xw_i)
        print('matrix inversion at w', time.process_time()-t0)

    if nops == 1:
        return intensities[0]
    else:
        return intensities

#You need to multiply by the 1/w^2 factor when plotting
def polarizability_corrvec_tda(td, freqs, fwhm=FWHM, BSE=False, orbs=None, imds_chkfile=None):
    kpts = td._scf.kpts
    #mo_coeff = np.array(td._scf.mo_coeff)
    #mo_occ = td._scf.mo_occ[0]
    #orbo = mo_coeff[:,:,mo_occ==2]
    #orbv = mo_coeff[:,:,mo_occ==0]
    mol = td._scf.mol

    mf_nocc = td._scf.mol.nelectron//2
    if orbs is None:
        mf_nmo = len(td._scf.mo_coeff[0][0,:])
        orbs = [x for x in range(mf_nmo)]
    nocc = sum([x < mf_nocc for x in orbs])
    nmo = len(orbs)
    nvir = nmo - nocc
    mo_coeff = np.array(td._scf.mo_coeff)[:,:,mf_nocc-nocc:mf_nocc+nvir]
    nkpts = np.shape(mo_coeff)[0]
    mo_occ = td._scf.mo_occ[0][mf_nocc-nocc:mf_nocc+nvir]
    orbo = mo_coeff[:,:,mo_occ==2]
    orbv = mo_coeff[:,:,mo_occ==0]

    #hermi=2 means anti-hermitian
    #maybe hermi=0 for pbc. meaning no symm. not sure.
    dipoles_ao = -1j*np.array(mol.pbc_intor('int1e_ipovlp', kpts=kpts, comp=3))
    nao = dipoles_ao.shape[-1]
    dipoles_ov = lib.einsum('kxpq,kpi,kqa->kxia', dipoles_ao.reshape(-1,3,nao,nao),
                            orbo.conj(), orbv)
    #This correction for GW non-locality only holds for non-hybrid functionals
    from pyscf.dft.libxc import is_hybrid_xc
    if not is_hybrid_xc(tdobj._scf.xc):
        td._scf.mo_energy = np.array(td._scf.mo_energy)
        td.mo_energy = np.array(td.mo_energy)
        moeocc = (td._scf.mo_energy[:,mf_nocc-nocc:mf_nocc+nvir])[:,mo_occ == 2]
        moevir = (td._scf.mo_energy[:,mf_nocc-nocc:mf_nocc+nvir])[:,mo_occ == 0]
        moediff = -moeocc[:,:,None] + moevir[:,None]
        tdmoeocc = (td.mo_energy[:,mf_nocc-nocc:mf_nocc+nvir])[:,mo_occ == 2]
        tdmoevir = (td.mo_energy[:,mf_nocc-nocc:mf_nocc+nvir])[:,mo_occ == 0]
        tdmoediff = -tdmoeocc[:,:,None] + tdmoevir[:,None]
        dipoles_ov = lib.einsum('kxij, kij->kxij', dipoles_ov, tdmoediff/moediff)
    nkpts, dim, nocc, nvir = dipoles_ov.shape
    dipoles_ov = dipoles_ov.transpose(1,0,2,3).reshape((dim,-1))

    intensities = np.zeros((3,len(freqs)))
    if BSE:
        if orbs is None:
            orbs = [x for x in range(td.mf_nmo)] 
        if os.path.isfile(imds_chkfile):
            qkLij = chkfile.load(imds_chkfile, 'qkLij')
            qeps_body_inv = chkfile.load(imds_chkfile, 'qeps_body_inv')
            all_kidx = chkfile.load(imds_chkfile, 'all_kidx')
        else:
            qkLij, qeps_body_inv, all_kidx = kbse.make_imds(td.gw, orbs)
            chkfile.dump(imds_chkfile, 'all_kidx', all_kidx)
            chkfile.dump(imds_chkfile, 'qkLij', qkLij)
            chkfile.dump(imds_chkfile, 'qeps_body_inv', qeps_body_inv)
        hdiag = td.get_diag(qkLij[0,:], qeps_body_inv[0], orbs)
        vind = lambda vec: kbse.matvec(td, np.array(vec, dtype='complex128'), qkLij, qeps_body_inv, all_kidx, orbs)
    else:
        vind = lambda vec: td.matvec(np.array(vec, dtype='complex128'), kshift=0)
        hdiag = td.get_diag(kshift=0)
        #vind, hdiag = td.gen_vind(td._scf, kshift=0)

    intensities = spectrum_corrvec(vind, dipoles_ov, freqs, hdiag=hdiag, fwhm=fwhm)

    #Xiao plots abs, not eps2, with 1/freqs
    #intensities /= freqs**2

    spin_fac = 2.0
    return spin_fac * intensities
