from pyphf import suscf, jk, sudft, sudm, util2
from pyphf.timing import timing
from pyscf import dft
import pyscf.dft.numint as numint
from automr import mcpdft
from automr.mcpdft import sum_adm2
import numpy as np
from functools import partial
#import time

try:
    from mrh.my_pyscf.mcpdft.mcpdft import _PDFT
    from mrh.util.rdm import get_2CDM_from_2RDM, get_2CDMs_from_2RDMs
    from mrh.my_pyscf.mcpdft.otfnal import energy_ot as get_E_ot
    #from mrh.my_pyscf.mcpdft.otfnal import transfnal, ftransfnal, get_transfnal
except:
    print('Warning: mrh not found')
print = partial(print, flush=True)
einsum = partial(np.einsum, optimize=True)


@timing
def kernel(pdft, suhf):
    if pdft.xc is None:
        if pdft.dens == 'dd':
            pdft.xc = 'pbe'
        else:
            pdft.xc = 'tpbe'
    print('\n******** %s ********' % pdft.__class__)
    print('method: SU%s-%s' % (pdft.dens.upper(), pdft.xc.upper()))
    mol = suhf.mol
    dm1 = suhf.suhf_dm
    print('energy decomposition')
    if suhf.debug:
        old_decomp(suhf, dm1)
    res = new_decomp(suhf, dm1)
    pdft.res = res
    if pdft.dens == 'dd':
        dmdefm = suhf.dm_reg
        #grids = sudft.set_grids(mol)
        grids = pdft.grids
        ni = numint.NumInt()
        n, exc, vxc = ni.nr_uks(mol, grids, pdft.xc, dmdefm)
        print('E_xcdft %.6f' % exc)
        if pdft.testd:
            n2, exc2, vxc2 = ni.nr_uks(mol, grids, pdft.xc, dm1)
            print('E_xcrho %.6f' % exc2)
            natorb = suhf.natorb[2]
            natocc = suhf.natocc[2]
            ua = 0.5*(natocc + natocc**2 * (2.0 - natocc))
            ub = 0.5*(natocc - natocc**2 * (2.0 - natocc))
            dm_ua = einsum('ij,j,kj -> ik', natorb, ua, natorb)
            dm_ub = einsum('ij,j,kj -> ik', natorb, ub, natorb)
            n3, exc3, vxc3 = ni.nr_uks(mol, grids, pdft.xc, (dm_ua, dm_ub))
            print('E_xcu   %.6f' % exc3)
    elif pdft.dens == 'pd':
        #pdft._init_ot_grids(pdft.xc)
        res = pdft.get_pd(suhf, pdft.otfnal, pdft.usemo, pdft.do_split)
        pdft.res = res

def check_2pdm(adm2s, dm1s, suhf):
    na = adm2s[0].shape[0]
    for i in range(na):
        for j in range(i,na):
            for k in range(na):
                for l in range(k,na):
                    if abs(adm2s[0][i,l,j,k]) > 1e-4:
                        print("aa %d %d %d %d %.6f" % (i,j,k,l,adm2s[0][i,l,j,k]))
                    if abs(adm2s[1][i,l,j,k]) > 1e-4:
                        print("ab %d %d %d %d %.6f" % (i,j,k,l,adm2s[1][i,l,j,k]))
                    if abs(adm2s[2][i,l,j,k]) > 1e-4:
                        print("bb %d %d %d %d %.6f" % (i,j,k,l,adm2s[2][i,l,j,k]))
    mol = suhf.mol
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    g = mol.intor("int2e")
    print(dm1s[0])
    e = einsum("pq, qp ->", h, 2*dm1s[0]) + 0.5 * einsum("pqrs, qrps ->", g, 4*(adm2s[0] + adm2s[1] + adm2s[2])) + mol.energy_nuc()
    print('redo e: %.6f' % e)
    
@timing
def get_pd(pdft, suhf, ot, usemo, do_split):
    #ot = _init_ot_grids (ot, suhf.mol)
    if do_split:
        xfnal, cfnal = ot.split_x_c()
    dm1s = np.array(suhf.suhf_dm)
    if usemo:
        _, [core, act, ext] = util2.dump_occ(suhf.natocc[2], 2.0, 0.99999)
        act_idx = slice(core, core+act)
        adm1s, adm2s = sudm.make_rdm12_no_native(suhf)
        adm1s = adm1s[:,act_idx, act_idx]
        #adm2s = adm2s[:,act_idx, act_idx, act_idx, act_idx]
        print(adm1s.shape, adm2s.shape)
    else:
        adm1s = dm1s
        adm2s = suhf.suhf_2pdm
        if suhf.debug2:
            check_2pdm(adm2s, dm1s, suhf)
        adm2s = adm2s[0].transpose(0,3,1,2)*2.0, adm2s[1].transpose(0,3,1,2)*2.0, adm2s[2].transpose(0,3,1,2)*2.0
    adm2 = sum_adm2(adm2s)
    #mo = suhf.natorb[2]
    if usemo:
        #mo = suhf.natorb[2][:,act_idx]
        mo = suhf.natorb[2]
    else:
        mo = np.eye(dm1s[0].shape[1])
    #dm1s = np.dot (adm1s, mo.T)
    #dm1s = np.dot (mo, dm1s).transpose (1,0,2)
    #print(dm1s)
    #dm1s += np.dot (mo_core, moH_core)[None,:,:]
    res = pdft.res
    if do_split:
        E_otx =  get_E_ot(xfnal, adm1s, adm2, mo, core)
        E_otc =  get_E_ot(cfnal, adm1s, adm2, mo, core)
        print('E_otx  : %15.8f' %E_otx)
        print('E_otc  : %15.8f' %E_otc)
        E_ot = E_otx + E_otc
        print('E_ot   : %15.8f' %E_ot)
        #return E_ot, E_otx, E_otc
        res['otx'] = E_otx
        res['otc'] = E_otc
        res['otxc'] = E_ot
    else:
        E_ot =  get_E_ot(ot, dm1s, adm2, mo)
        print('E_ot   : %15.8f' %E_ot)
        #return E_ot
        res['otxc'] = E_ot
    return res


def new_decomp(suhf, dm1):
    enuc = suhf.energy_nuc
    dm1t = dm1[0] + dm1[1]
    Ecore = np.trace(np.dot(suhf.hcore_reg, dm1t))
    vj, vk = jk.get_jk(suhf.mol, dm1)
    veffj = vj[0] + vj[1] 
    veffk = -vk
    Ej = np.trace(np.dot(veffj, dm1[0]) + np.dot(veffj, dm1[1])) * 0.5
    Ek = np.trace(np.dot(veffk[0], dm1[0]) + np.dot(veffk[1], dm1[1])) * 0.5
    Ejk = Ej + Ek
    Ec = suhf.E_suhf - enuc - Ecore - Ejk
    print('E_suhf : %15.8f' % suhf.E_suhf)
    print('E_nn   : %15.8f' % enuc)
    print('E_core : %15.8f' % Ecore)
    print('E_j    : %15.8f' % Ej)
    print('E_k    : %15.8f' % Ek)
    if suhf.debug:
        print('E_jk   : %15.8f' % Ejk)
    print('E_c    : %15.8f' % Ec)
    res = {'suhf': suhf.E_suhf,
           'j': Ej,
           'k': Ek,
           'c': Ec}
    return res

def old_decomp(suhf, dm1):
    dm1t = dm1[0] + dm1[1]
    enuc = suhf.energy_nuc
    print('E_nuc %.6f' % enuc)
    #omega, alpha, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc, spin=spin)
    Jg, Kg = suhf.get_JKg()
    #hfx = suhf.get_EX()
    E0, Ejk, E1j, E1k = get_H(suhf, suhf.hcore_ortho, suhf.no, suhf.Pg, suhf.Gg, Jg, Kg, suhf.xg)
    E_suhf = enuc + E0 + Ejk
    print('E_suhf %.6f' % E_suhf)

def get_H(suhf, hcore_ortho, no, Pg, Gg, Jg, Kg, xg):
    #print(hcore_ortho)
    hcore_ortho = np.vstack((
        np.hstack((hcore_ortho, np.zeros(hcore_ortho.shape))),
        np.hstack((np.zeros(hcore_ortho.shape), hcore_ortho))
    ))
    hcore_no = einsum('ji,jk,kl->il', no, hcore_ortho, no)
    suhf.hcore_no = hcore_no
    if suhf.debug2:
        print(hcore_no)
    trHg0 = np.zeros(len(Pg))
    trHg1 = np.zeros(len(Pg))
    trHg1j = np.zeros(len(Pg))
    trHg1k = np.zeros(len(Pg))

    for i, pg in enumerate(Pg):
        H0 = np.trace(np.dot(hcore_no, pg)) 
        H1 = 0.5 * np.trace(np.dot(Gg[i], pg))
        H1j = 0.5 * np.trace(np.dot(Jg[i], pg))
        H1k = 0.5 * np.trace(np.dot(Kg[i], pg))
        #H = H * xg[i]
        trHg0[i] = H0 
        trHg1[i] = H1
        trHg1j[i] = H1j
        trHg1k[i] = H1k
        #print(i, H*xg[i])
    H0 = suhf.integr_beta(trHg0, fac='xg')
    H1 = suhf.integr_beta(trHg1, fac='xg')
    H1j = suhf.integr_beta(trHg1j, fac='xg')
    H1k = suhf.integr_beta(trHg1k, fac='xg')
    #H0 = suhf.integr_beta(trHg, fac='xg')
    #print('H ', H0, H1, H1j, H1k)
    print('E_core : %15.8f' % H0)
    print('E_jk   : %15.8f' % H1)
    print('E_j    : %15.8f' % H1j)
    print('E_k    : %15.8f' % H1k)
    #print('ciH', ciH0, ciH1, ciH1j, ciH1k)
    #suhf.trHg = trHg
    return H0, H1, H1j, H1k

class PDFT(mcpdft.PDFT):
    def __init__(self, suhf, xc, dens, grids_level=4):
        self.suhf = suhf
        self.mol = suhf.mol
        self.verbose = 5
        self.stdout = suhf.stdout
        #if xc is not None:
        self.xc = xc
        #else:
        #    self.xc = None
        self.dens = dens
        self.testd = False
        self.usemo = True
        self.do_split = False
        self.grids_level = grids_level

    def kernel(self):
        if self.dens == 'pd':
            self._init_ot(self.xc)
        grids_attr = {}
        if self.grids_level is not None:
            grids_attr['level'] = self.grids_level
        self._init_grids(grids_attr)
        return kernel(self, self.suhf)
    
    get_pd = get_pd