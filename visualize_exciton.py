import numpy as np
from pyscf import lib
from pyscf.tools import cubegen
from pyscf.pbc import gto, scf, tdscf
import kbse as bse


RESOLUTION = None
BOX_MARGIN = 0.0


def orbital_k(cell, supercell, outfile, kpt, coeff, nx=80, ny=80, nz=80,
              resolution=RESOLUTION, margin=BOX_MARGIN):
    """Calculate orbital value on real space grid and write out in cube format.

    Args:
        mol : Mole
            Molecule to calculate the electron density for.
        outfile : str
            Name of Cube file to be written.
        coeff : 1D array
            coeff coefficient.

    Kwargs:
        nx : int
            Number of grid point divisions in x direction.
            Note this is function of the molecule's size; a larger molecule
            will have a coarser representation than a smaller one for the
            same value. Conflicts to keyword resolution.
        ny : int
            Number of grid point divisions in y direction.
        nz : int
            Number of grid point divisions in z direction.
        resolution: float
            Resolution of the mesh grid in the cube box. If resolution is
            given in the input, the input nx/ny/nz have no effects.  The value
            of nx/ny/nz will be determined by the resolution and the cube box
            size.
    """
    cc = cubegen.Cube(supercell, nx, ny, nz, resolution, margin)

    GTOval = 'PBCGTOval'

    # Compute density on the .cube grid
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    orb_on_grid = np.empty(ngrids, dtype=complex)
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao = cell.eval_gto(GTOval, coords[ip0:ip1], kpt=kpt)  # note kpt
        orb_on_grid[ip0:ip1] = np.dot(ao, coeff)
    orb_on_grid = orb_on_grid.reshape(cc.nx,cc.ny,cc.nz)

    # Write out orbital to the .cube file
    #cc.write(orb_on_grid.real, outfile, comment='Orbital value in real space (1/Bohr^3)')
    return orb_on_grid


def main():

    from pyscf.pbc.tools import pyscf_ase, lattice
    formula = 'LiF'
    basis = 'gth-szv'
    ase_atom = lattice.get_ase_atom(formula)
    cell = gto.Cell()
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.a = ase_atom.cell
    cell.unit = 'B'
    cell.basis = basis
    cell.pseudo = 'gth-pade'
    # cell.precision = 5e-10
    cell.build()

    kpt_mesh = [3,3,3]
    kpts = cell.make_kpts(kpt_mesh)
    from pyscf.pbc import dft
    kmf = dft.KRKS(cell, exxdiv=None).density_fit()
    kmf.kpts = kpts
    kmf.xc = 'LDA'
    kmf.kernel()
    
    nocc = cell.nelectron //2
    # nmo = cell.nao_nr()

    # Periodic TDA
    td = tdscf.TDA(kmf)
    es, xys = td.kernel()
    #Q=0
    es = es[0]
    xys = xys[0]
    
    # Periodic BSE
    #TODO: update to work with the BSE I implemented for PySCF
    # from pyscf.pbc.gw import krgw_ac
    # lowest_gw_occ = max(0,nocc - 3)
    # highest_gw_vir = min(nmo, nocc + 6)
    # orbs = range(lowest_gw_occ, highest_gw_vir)
    # mygw = krgw_ac.KRGWAC(kmf)
    # mygw.linearized = False
    # mygw.ac = 'pade'
    # mygw.fc = False
    # nocc = mygw.nocc
    # mygw.kernel(orbs=orbs, nw=80)
    # mytd = bse.BSE(mygw, TDA=True, singlet=True, CIS=False)
    # conv, nstates, es, xys = mytd.kernel(orbs=orbs, nstates=4)
    
    # First three roots
    exc = 0
    degen = 3

    from pyscf.pbc.tools import super_cell
    scell = super_cell(cell, kpt_mesh)
    scell.build()

    nx = 20
    ny = 20
    nz = 20
    from pyscf.tools import cubegen
    cc = cubegen.Cube(scell, nx, ny, nz)
    coords = cc.get_coords()
    hole_pos = [nx//2+1, ny//2+1, nz//2+1]  # shift slightly off atom to avoid nodes
    # hole = 1.0/np.linalg.norm(coords-coords.reshape(nx,ny,nz,3)[*hole_pos], axis=1)
    # hole = hole.reshape([nx, ny, nz])
    hole = np.zeros((nx, ny, nz))
    hole[*hole_pos] = 400
    cc.write(hole, f'hole_{kpt_mesh[0]}{kpt_mesh[1]}{kpt_mesh[2]}.cub')  #visualize hole location

    density = np.zeros((nx, ny, nz))
    for exc in range(exc+degen): #threefold degen LiF
        wfn = np.zeros((nx, ny, nz), dtype=complex)
        x = xys[exc][0] #X
        for k, kpt in enumerate(kpts):
            xk = x[k]
            _nocc, _nvir = xk.shape
            for i in range(_nocc):
                psik_i = orbital_k(cell, scell, f'orb_nk{k}_occ-{i}-{basis}.cub', kpt, kmf.mo_coeff[k][:, nocc-_nocc:][:,i], nx, ny, nz)[*hole_pos].conj()
                for a in range(_nvir):
                    wfn += xk[i,a] * psik_i \
                        * orbital_k(cell, scell, f'orb_nk{k}_vir-{a}-{basis}.cub', kpt, kmf.mo_coeff[k][:,nocc+a], nx, ny, nz)
        density += np.abs(wfn)**2
    cc = cubegen.Cube(scell, nx, ny, nz)
    cc.write(density, f'exc_wfn_{kpt_mesh[0]}{kpt_mesh[1]}{kpt_mesh[2]}-{basis}.cub')
    
    
    dV = scell.vol/(nx*ny*nz) #volume in Bohr
    print('V', scell.vol)
    print('dV', dV)
    dV = cc.get_volume_element()
    print('dV', dV)
    total_density = np.sum(density) #maybe doesn't add up to 1 bc hole is fixed?
    print('total density', total_density)
    sigmas = np.linspace(1e-05, 3e-5, 10) 
    integrals = [np.sum([d for d in density.ravel() if d > sigma]) for sigma in sigmas]
    print('sigmas', sigmas)
    print('integrals', integrals/total_density)


if __name__ == '__main__':
    main()
