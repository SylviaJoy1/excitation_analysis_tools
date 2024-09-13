import numpy as np
import scipy
import time
from pyscf import lib

FWHM = 0.2/27.21139

import sys
sys.path.append('/burg/berkelbach/users/sjb2225/v2.4.0/bse_master/static_BSE/periodic')
import kbse

def lorentzian(x, mean, fwhm):
    eta = fwhm/2 #https://mathworld.wolfram.com/LorentzianFunction.html#:~:text=The%20Lorentzian%20function%20gives%20the,Lorentzian%20function%20has%20Fourier%20transform
    return 1.0/np.pi * eta / ((x-mean)**2 + eta**2)

def _contract_multipole(tdobj, ints, BSE=False, hermi=True, xy=None, orbs=None):
    if xy is None: xy = tdobj.xy
    
    mf_nocc = tdobj._scf.mol.nelectron//2
    if orbs is None:
        mf_nmo = len(tdobj._scf.mo_coeff[0][0,:])
        orbs = [x for x in range(mf_nmo)]
    nocc = sum([x < mf_nocc for x in orbs])
    nmo = len(orbs)
    nvir = nmo - nocc

    mo_coeff = np.array(tdobj._scf.mo_coeff)[:,:,mf_nocc-nocc:mf_nocc+nvir]
    nkpts = np.shape(mo_coeff)[0]
    mo_occ = tdobj._scf.mo_occ[0][mf_nocc-nocc:mf_nocc+nvir]
    orbo = mo_coeff[:,:,mo_occ==2]
    orbv = mo_coeff[:,:,mo_occ==0]

    if not BSE:
        xy = xy[0]
        
    nstates = len(xy)
    
    pol_shape = ints.shape[1:-2]
 
    nao = ints.shape[-1]

    ints = lib.einsum('kxpq,kpi,kqj->kxij', ints.reshape(nkpts, -1,nao,nao), orbo.conj(), orbv)
    #This correction for GW non-locality only holds for non-hybrid functionals
    from pyscf.dft.libxc import is_hybrid_xc
    if not is_hybrid_xc(tdobj._scf.xc):
        moeocc = tdobj._scf.mo_energy[:,mf_nocc-nocc:mf_nocc+nvir][:, mo_occ == 2]
        moevir = tdobj._scf.mo_energy[:,mf_nocc-nocc:mf_nocc+nvir][:, mo_occ == 0]
        moediff = -moeocc[:,None] + moevir
        tdmoeocc = tdobj.mo_energy[:,mf_nocc-nocc:mf_nocc+nvir][:, mo_occ == 2]
        tdmoevir = tdobj.mo_energy[:,mf_nocc-nocc:mf_nocc+nvir][:, mo_occ == 0]
        tdmoediff = -tdmoeocc[:,None] + tdmoevir
        ints = lib.einsum('kxij, kij->kxij', ints, tdmoediff/moediff)
    
    pol = []
    for nk in range(nkpts):
        _pol = np.array([np.einsum('xij,ij->x', ints[nk], x[nk]) * 2 for x,y in xy])
        #only matters for TDDFT
        if isinstance(xy[0][1], np.ndarray): #check if Y is an array, not just 0 float
            raise NotImplementedError("TODO: reshape x,y array correctly for H op")
            # if hermi:
            #     _pol += [np.einsum('xij,ij->x', ints[nk], np.array(y)) * 2 for x,y in xy]
            # else:  # anti-Hermitian
            #     _pol -= [np.einsum('xij,ij->x', ints[nk], np.array(y)) * 2 for x,y in xy]
        pol.append(_pol)
    pol = np.array(pol).reshape((nkpts, nstates,)+pol_shape)
    return pol

def transition_velocity_dipole(tdobj, BSE=False, orbs=None, xy=None):
    '''Transition dipole moments in the velocity gauge (imaginary part only)
    hence the 'i' in 'ipovlp', I guess
    '''
    kpts = tdobj._scf.kpts

    ints = -1j*np.array(tdobj._scf.mol.pbc_intor('int1e_ipovlp', kpts=kpts, comp=3))
    v = _contract_multipole(tdobj, ints, BSE, hermi=False, xy=xy, orbs=orbs)
    return v #removed the -

def polarizability_tda(td, freqs, BSE=False, fwhm=FWHM, orbs=None, lineshape_fn=None): 
    nkpts = np.shape(td._scf.mo_coeff)[0]
    
    if lineshape_fn is None:
        lineshape_fn = lorentzian
    
    #shape nkpts, nexc, 3
    trans_dip = transition_velocity_dipole(td, BSE, orbs=orbs)
    
    if not BSE:
        es = td.e[0]
    else:
        es = td.e
       
    ls = []   
    for n, en in enumerate(es):    #for kshift 0
        ls.append(lineshape_fn(freqs, en, fwhm))
    
    intensities = np.einsum('nx, nw->xw', np.abs(np.einsum('knx->nx', trans_dip))**2, ls)

    return intensities.real#/freqs**2

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
            xw_i, exit_code = scipy.sparse.linalg.gcrotmk(A, ops[i].conj(), x0=x0, M=M, tol=1e-1)
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
def polarizability_corrvec_tda(td, freqs, fwhm=FWHM, BSE=False, orbs=None):
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
    #only for non-hybrid functionals
    moeocc = td._scf.mo_energy[:,mf_nocc-nocc:mf_nocc+nvir][:, mo_occ == 2]
    moevir = td._scf.mo_energy[:,mf_nocc-nocc:mf_nocc+nvir][:, mo_occ == 0]
    moediff = -moeocc[:,None] + moevir
    tdmoeocc = td.mo_energy[:,mf_nocc-nocc:mf_nocc+nvir][:, mo_occ == 2]
    tdmoevir = td.mo_energy[:,mf_nocc-nocc:mf_nocc+nvir][:, mo_occ == 0]
    tdmoediff = -tdmoeocc[:,None] + tdmoevir
    dipoles_ov = lib.einsum('kxij, kij->kxij', dipoles_ov, tdmoediff/moediff)
    nkpts, dim, nocc, nvir = dipoles_ov.shape
    dipoles_ov = dipoles_ov.transpose(1,0,2,3).reshape((dim,-1))

    intensities = np.zeros((3,len(freqs)))
    if BSE:
        #if orbs is None:
        #    orbs = [x for x in range(td.mf_nmo)] 
        qkLij, qeps_body_inv, all_kidx = kbse.make_imds(td.gw, orbs)
        hdiag = td.get_diag(qkLij[0,:], qeps_body_inv[0], orbs)
        vind = lambda vec: kbse.matvec(td, np.array(vec, dtype='complex128'), qkLij, qeps_body_inv, all_kidx, orbs)
    else:
        vind = lambda vec: td.matvec(vec, kshift=0)
        hdiag = td.get_diag(kshift=0)
        #vind, hdiag = td.gen_vind(td._scf, kshift=0)

    intensities = spectrum_corrvec(vind, dipoles_ov, freqs, hdiag=hdiag, fwhm=fwhm)

    #intensities /= freqs**2

    spin_fac = 2.0
    return spin_fac * intensities
