from pyscf import lib
from pyphf import suscf, supdft
from automr import guess, util

lib.num_threads(4)

xyz = '''N 0.0 0.0 0.0; N 0.0 0.0 1.097'''
bas = 'cc-pvtz'

mf = guess.from_frag(xyz, bas, [[0],[1]], [0,0], [3,-3])

# SU-rsDFT
mf2 = suscf.SUHF(mf)
mf2.dft = True
mf2.xc = util.get_xc2('rsblyp', omega=0.4, srdft=0.5)
mf2.kernel()
mf2 = mf2.to_hf()
mf2.guesshf.mo_coeff = mf2.mo_reg
mf2.noiter=True
mf2.kernel()

mf3 = supdft.PDFT(mf2, 'tblyp', 'pd')
mf3.kernel()


