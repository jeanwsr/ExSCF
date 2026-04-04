# ExSCF
ExSCF is a program for methods based on Projected Hartree-Fock.

## Pre-requisites
* sympy
* [PySCF](https://github.com/pyscf/pyscf)
* [pyAutoMR](https://github.com/hebrewsnabla/pyAutoMR)
* [MOKIT](https://gitlab.com/jxzou/mokit) (optional, for read/write fch)

Of course we need numpy, scipy, etc., but these are also required by PySCF, so they are not listed here.

## Installation
(1) From pypi
```
pip install exscf
```
(2) From source
* git clone
* add `/path/to/ExSCF` to your `PYTHONPATH`

## Features
**Theoretical features**:
* SUHF (Spin-projected Unrestricted Hartree-Fock)
  + energy
  + 1pdm, natural orbitals, 2pdm
* SU-PDFT

 not fully tested, or not fully implemented yet:
* SUHF+DFT, SUHF+*f*DFT, SUHF+*fc*DFT
* CAS+*f*DFT, CAS-DFT2
* DeltaSCF with MOM
* PAV-SUMP2, SUMP2(0)

**Technical features**:
* guess
  + mix (`guess.mix`)
  + fragment (`guess.from_frag`)
  + stablize UHF (`guess.check_stab`)
* converging strategy
  + DIIS (`diis_on = True`)
  + `level_shift = ` (in a.u.)
* interface (require MOKIT)
  + read guess MO from fch (`guess.from_fch_simp`)
  + dump MO, NO to fch (`tofch = True`)

more guess strategies are included in [pyAutoMR](https://github.com/hebrewsnabla/pyAutoMR).

## Quick Start
```
from pyphf import suscf, guess

xyz = 'H 0.0 0.0 0.0; H 0.0 0.0 2.0'''
bas = '3-21g'
mf = guess.mix(xyz, bas, conv='tight')

mf2 = suscf.SUHF(mf)
mf2.kernel()
```

## Contact
For bug report or comments, please contact the author via srwang20@fudan.edu.cn or open an issue.

