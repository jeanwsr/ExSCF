from pyphf import suscf, supdft, sudft, symm

__version__ = '0.5.0'

def SUHF(mf):
    if not mf.mol.symmetry or mf.mol.groupname=='C1':
        return suscf.SUHF(mf)
    else:
        return symm.SymAdaptedSUHF(mf)

SUDFT  = sudft.SUDFT
