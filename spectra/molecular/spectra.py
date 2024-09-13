import numpy
import scipy

from pyscf import lib
from pyscf.tdscf.rhf import _charge_center

FWHM = 0.2/27.21139


def lorentzian(x, mean, fwhm):
    eta = fwhm/2
    return 1.0/numpy.pi * eta / ((x-mean)**2 + eta**2)


def polarizability_tda(td, freqs, fwhm=FWHM, gauge='length', lineshape_fn=None): 
    if lineshape_fn is None:
        lineshape_fn = lorentzian
    if gauge == 'length':
        trans_dip = td.transition_dipole()
    else:
        trans_dip = td.transition_velocity_dipole()
        trans_dip = numpy.einsum('s,sx->sx', 1./td.e, trans_dip)
    intensities = numpy.zeros((3,len(freqs)))
    for n, en in enumerate(td.e):
        ls = lineshape_fn(freqs, en, fwhm)
        for x in range(3):
            intensities[x] += trans_dip[n][x]**2 * ls 

    return intensities


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

    if len(numpy.array(ops).shape) == 2:
        nops, size = numpy.array(ops).shape
    else:
        nops, size = 1, len(ops)
        ops = list(ops)

    intensities = numpy.zeros((nops,len(freqs)))

    for w, freq in enumerate(freqs):
        def matvec(x):
            hx = hop(x)
            return (freq**2 + eta**2)*x - 2*freq*hx + hop(hx)

        A = scipy.sparse.linalg.LinearOperator((size,size), matvec=matvec)

        M = None
        if hdiag is not None:
            def precond(x):
                return x/(freq**2 - 2*freq*hdiag + hdiag*hdiag + eta**2)
            M = scipy.sparse.linalg.LinearOperator((size,size), matvec=precond)

        for i in range(nops):
            x0 = precond(ops[i])
            xw_i, exit_code = scipy.sparse.linalg.gcrotmk(A, ops[i], x0=x0, M=M)
            #xw_i, exit_code = scipy.sparse.linalg.gmres(A, ops[i], x0=x0, M=M)

            if exit_code != 0:
                print(f"Warning! GCROT(m,k) not converged after {exit_code} iterations")
            
            intensities[i,w] = 1/numpy.pi * eta * numpy.dot(ops[i].conj(), xw_i)

    if nops == 1:
        return intensities[0]
    else:
        return intensities


def polarizability_corrvec_tda(td, freqs, fwhm=FWHM, gauge='length'):
    mo_coeff = td._scf.mo_coeff
    mo_occ = td._scf.mo_occ
    orbo = mo_coeff[:,mo_occ==2]
    orbv = mo_coeff[:,mo_occ==0]
    mol = td.mol
    if gauge == 'length':
        with mol.with_common_orig(_charge_center(mol)):
            dipoles_ao = mol.intor_symmetric('int1e_r', comp=3)
    else:
        dipoles_ao = mol.intor('int1e_ipovlp', comp=3, hermi=2)
    nao = dipoles_ao.shape[-1]
    dipoles_ov = lib.einsum('xpq,pi,qa->xia', dipoles_ao.reshape(3,nao,nao),
                            orbo.conj(), orbv)
    _, nocc, nvir = dipoles_ov.shape
    dipoles_ov = dipoles_ov.reshape((3,-1))

    intensities = numpy.zeros((3,len(freqs)))
    vind, hdiag = td.gen_vind()

    intensities = spectrum_corrvec(vind, dipoles_ov, freqs, hdiag=hdiag, fwhm=fwhm)

    if gauge != 'length':
        intensities /= freqs**2

    spin_fac = 2.0
    return spin_fac * intensities


