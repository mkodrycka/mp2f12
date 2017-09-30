from psi4 import get_global_option
from psi4 import MintsHelper, BasisSet, OrbitalSpace, Matrix
import numpy as np

np.set_printoptions(precision=6, linewidth=500, suppress=True)


def option(keyword):
    """
    Returns the requested option from the psi4 input file.
    """
    return get_global_option(keyword.upper())


def mintshelper(molecule):
    """
    Returns a psi4 MintsHelper object
    """
    basisname = option('basis')
    basisset = BasisSet.pyconstruct_orbital(molecule, 'BASIS', basisname, True)

    return MintsHelper(basisset)


def basisset(molecule):
    """
    Returns the BasisSet as requested in the psi4 input file.
    """
    basisname = option('basis')
    
    return BasisSet.pyconstruct_orbital(molecule, 'BASIS', basisname, True)


def get_cabs(molecule, obs, lindep = 1E-9):
    """
    Returns the CABS space
    """
    ribs = get_ribs(molecule, lindep)
    cabs = OrbitalSpace.build_cabs_space(obs, ribs, lindep)

    return cabs


def get_ribs(molecule, lindep = 1E-9):
    """
    Returns the RIBS.
    """
    ribs = OrbitalSpace.build_ri_space(molecule, 'BASIS', 'DF_BASIS_MP2', lindep)

    return ribs


def orbital_spaces(molecule, nocc, C, cabs_flag = False, lindep = 1E-9):
    """
    Returns a dictionary of OrbitalSpaces.
    C is the full OBS MO coefficient matrix.
    """
    ints = mintshelper(molecule).integral()
    basis = basisset(molecule)
    nfocc = nfzc(molecule)

    bs = {}
    bs['focc'] = orbital_space('f', 'FOCC', C[:,:nfocc], basis, ints) 
    bs['aocc'] = orbital_space('a', 'AOCC', C[:,nfocc:nocc], basis, ints)
    bs['occ'] = orbital_space('o', 'OCC', C[:,:nocc], basis, ints)
    bs['vir'] = orbital_space('v', 'VIR', C[:,nocc:], basis, ints)
    bs['act'] = orbital_space('r', 'ACT', C[:,nfocc:], basis, ints)
    bs['obs'] = orbital_space('p', 'OBS', C, basis, ints)
    if cabs_flag:
        bs['cabs'] = get_cabs(molecule, bs['obs'], lindep)

    return bs
    

def orbital_space(osid, osname, C, basis, integrals):
    """
    Returns the requested orbital space.
    """
    nrows, ncols = C.shape
    Cmat = Matrix(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            Cmat.set(i, j, C[i,j])

    return OrbitalSpace(osid, osname, Cmat, basis, integrals)


def nfzc(molecule):
    """
    Returns the number of frozen core orbitals.
    """
    nfzc = 0
    
    if option('freeze_core') == 'TRUE':
        for i in range(molecule.natom()):
            if molecule.Z(i) > 2: nfzc += 1
            if molecule.Z(i) > 10: nfzc += 4
            if molecule.Z(i) > 18: nfzc += 4
            if molecule.Z(i) > 36: nfzc += 9
            if molecule.Z(i) > 54: nfzc += 9
            if molecule.Z(i) > 86: nfzc += 16
            if molecule.Z(i) > 108:
                raise Exception('Invalid atomic number.')

    return nfzc

