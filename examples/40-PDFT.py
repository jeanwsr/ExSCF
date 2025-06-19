from pyscf import lib
from pyphf import suscf, guess, supdft

xyz = '''N 0.0 0.0 0.0; N 0.0 0.0 2.0'''
bas = 'cc-pvdz'

lib.num_threads(8)

mf = guess.from_frag(xyz, bas, [[0],[1]], [0,0], [3,-3])
mf2 = suscf.SUHF(mf)
#mf2.do2pdm = True
mf2.kernel()

mf3 = supdft.PDFT(mf2, 'tpbe', 'pd')
mf3.kernel()

mf4 = supdft.PDFT(mf2, 'pbe', 'dd')
mf4.kernel()

