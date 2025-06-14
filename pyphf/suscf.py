import numpy as np
import scipy
from pyscf import scf, lib
from pyscf.scf import chkfile
from pyscf.dft import numint
#from pyscf.lib.misc import repo_info
from pyphf import sudm, sudft, deltascf
from pyphf import util2, jk, wigner
import os
from functools import partial
import time

print = partial(print, flush=True)
einsum = partial(np.einsum, optimize=True)

'''
SUHF
Jiménez-Hoyos, C. A.; Henderson, T. M.; Tsuchimochi, T.; Scuseria, G. E. J Chem Phys 2012, 136, 164109
'''

def eig(A):
    return np.linalg.eigh(A)
    #return scipy.linalg.eigh(A)

def get_occ(suhf, mo_e=None, mo_coeff=None):
    na, nb = suhf.nelec
    nmo = suhf.norb
    mo_occ = np.zeros((2,nmo), dtype=int)
    if mo_e is None:
        mo_occ[0,:na] = 1
        mo_occ[1,:nb] = 1
    else:
        e_idx_a = np.argsort(mo_e[0])
        e_idx_b = np.argsort(mo_e[1])
        e_sort_a = mo_e[0][e_idx_a]
        e_sort_b = mo_e[1][e_idx_b]
        #nmo = mo_e[0].size
        mo_occ[0,e_idx_a[:na]] = 1
        mo_occ[1,e_idx_b[:nb]] = 1
    return mo_occ

def get_vir(occa, occb):
    vira = np.ones(len(occa), dtype=int) - occa
    virb = np.ones(len(occb), dtype=int) - occb
    return vira, virb

def count0(vals):
    non0 = 0
    for i in vals:
        if abs(i)>1e-10:
            non0 += 1
    return non0

def find_NO(suhf, dm, mo_occ):
    cut_no = suhf.cut_no
    #dm = dm*(-1)
    #print(dm)
    occa, occb = mo_occ
    vira, virb = get_vir(occa, occb)
    na, nb = suhf.nelec
    #print(occa,occb,vira, virb)
    ev_a, v_a = eig(dm[0]*(-1))
    ev_b, v_b = eig(dm[1]*(-1))
    pa = count0(ev_a)
    pb = count0(ev_b)
    if suhf.debug or suhf.printmo:
        print('NO eigenvalue')
        print(ev_a, '\n', ev_b)
#    v_a1 = v_a[:,occa == 1]
#    v_a2 = v_a[:,vira == 1]
#    v_b1 = v_b[:,occb == 1]
#    v_b2 = v_b[:,virb == 1]
    v_a1 = v_a[:,:na]
    v_a2 = v_a[:,na:]
    v_b1 = v_b[:,:nb]
    v_b2 = v_b[:,nb:]
    #print(v_a1, v_a2, v_b1, v_b2)
    v_a1 = np.vstack((v_a1, np.zeros(v_a1.shape)))
    v_a2 = np.vstack((v_a2, np.zeros(v_a2.shape)))
    v_b1 = np.vstack((np.zeros(v_b1.shape), v_b1))
    v_b2 = np.vstack((np.zeros(v_b2.shape), v_b2))
    #print(v_a1, v_a2, v_b1, v_b2)

    v = np.hstack((v_a1, v_b1, v_a2, v_b2))
    if cut_no:
        v = np.hstack((v_a1, v_b1, v_a2, v_b2))[:,:pa+pb]
    #v = np.hstack((v, np.zeros((v.shape[0], v.shape[0]-pa-pb))))
    if suhf.debug or suhf.printmo:
        print('NO vec')
        print(v)
    dm_expd = np.hstack(
        (np.vstack((dm[0], np.zeros(dm[0].shape))), 
        np.vstack((np.zeros(dm[1].shape), dm[1])))
        )
    #print(dm_expd)
    dm_no = einsum('ji,jk,kl->il', v, dm_expd, v)
    if suhf.debug:
        print('dm(NO)')
        print(dm_no)
    #np.set_printoptions(precision=6, linewidth=160, suppress=True)
    return dm_no, dm_expd, v

def get_Ng(grids, no, dm, occ):
    Dg = []
    Ng = []
    Pg = []
    #print('N(g)')
    for beta in grids:
        norb = int(len(no)/2)
        dg1 = np.eye(norb) * np.cos(beta/2)
        dg2 = np.eye(norb) * (-np.sin(beta/2))
        dg = np.vstack((
            np.hstack((dg1, dg2)),
            np.hstack((-dg2, dg1))
        ))
        #print(dg.shape)
        #print(dg)
        dg_no = einsum('ji,jk,kl->il', no, dg, no)
        #det_dg = np.linalg.det(dg_no)
        #print(det_dg)
        #print(dg_no)
        Dg.append(dg_no)
#        Dg.append(dg)
        tg = einsum('ij,jk,kl->il', dm[:occ,:], dg_no, dm[:,:occ])
        ng = np.linalg.inv(tg)
        #print(ng)
        Ng.append(ng)
        pg = einsum('ij,jk,kl,lm->im', dg_no, dm[:,:occ], ng, dm[:occ,:])
        #print(pg)
        Pg.append(pg)
    return Dg, Ng, Pg

def expd(mo, mo_occ):
    C_a, C_b = mo
    occa, occb = mo_occ
    vira, virb = get_vir(occa, occb)
    C_a1 = C_a[:,occa==1]
    C_a2 = C_a[:,vira==1]
    C_b1 = C_b[:,occb==1]
    C_b2 = C_b[:,virb==1]
    C_org = np.hstack((
        np.vstack((C_a1, np.zeros(C_a1.shape))),
        np.vstack((np.zeros(C_b1.shape), C_b1)),
        np.vstack((C_a2, np.zeros(C_a2.shape))),
        np.vstack((np.zeros(C_b2.shape), C_b2))
    ))
    return C_org

def get_xg(suhf, no, mo_occ, Ng):
    C_org = expd(suhf.mo_ortho, mo_occ)
    #print(C_org)
    C_no = einsum('ji,jk->ik', no, C_org)
    if suhf.debug:
        print('C(NO)')
        print(C_no)
    na,nb = suhf.nelec
    occ = na+nb
    C_oo = C_no[:occ, :occ]
    detC = np.linalg.det(C_oo)
    print('detC %.8f'%detC)
    detNg = []
    for ng in Ng:
        detng = np.linalg.det(ng)
        #print(detng)
        detNg.append(detng)
    detNg = np.array(detNg)
    print('detNg', detNg)
    xg = 1.0 / (detC * detNg * detC)
    print('xg', xg)
    ciS = suhf.integr_beta(xg)
    #print(weights, xg, d)
    print('ciS', ciS)
    yg = xg / ciS
    return xg, yg, ciS, C_no

def integr_beta(q, d, grids, weights, fac='normal', xg=None, ciS=None):
    sinbeta = np.sin(grids)
    if fac=='xg':
        weights = weights * xg / ciS
        #print(xg, ciS, xg/ciS**2)
        #weights *= (xg / ciS**2)
    elif fac=='ci':
        weights = weights / ciS

    if q.ndim==1:
        int_q = einsum('i,i,i,i->', weights, sinbeta, q, d)
    elif q.ndim==2: 
        int_q = einsum('i,i,ij,i->j', weights, sinbeta, q, d)
    elif q.ndim==3:
        int_q = einsum('i,i,ijk,i->jk', weights, sinbeta, q, d)
    elif q.ndim==5:
        int_q = einsum('i,i,ijklm,i->jklm', weights, sinbeta, q, d)
    else:
        raise(ValueError, 'q must have dimension 1, 2, 3 or 5')
    return int_q


def get_H(suhf, hcore_ortho, no, Pg, Gg, xg):
    #print(hcore_ortho)
    hcore_ortho = np.vstack((
        np.hstack((hcore_ortho, np.zeros(hcore_ortho.shape))),
        np.hstack((np.zeros(hcore_ortho.shape), hcore_ortho))
    ))
    hcore_no = einsum('ji,jk,kl->il', no, hcore_ortho, no)
    suhf.hcore_no = hcore_no
    if suhf.debug2:
        print(hcore_no)
    trHg = np.zeros(len(Pg))
    for i, pg in enumerate(Pg):
        H0 = np.trace(np.dot(hcore_no, pg)) 
        H1 = 0.5 * np.trace(np.dot(Gg[i], pg))
        #H = H * xg[i]
        trHg[i] = H0 + H1
        #print(i, H*xg[i])
    ciH = suhf.integr_beta(trHg*xg)
    print('ciH', ciH)
    suhf.trHg = trHg
    H = suhf.integr_beta(trHg, fac='xg')
    print('Hsp + Hph = ', H)
    return trHg, ciH, H

def get_E1(suhf, hcore_no, Pg):
    trHg = np.zeros(len(Pg))
    for i, pg in enumerate(Pg):
        H = np.trace(np.dot(hcore_no, pg))
        trHg[i] = H
        #print(i, H*xg[i])
    E1 = suhf.integr_beta(trHg, fac='xg')
    print('E1 = %.6f'% E1)
    return E1

def get_EX(suhf, no, Pg, Kg, xg):
    trXg = np.zeros(len(Pg))
    for i, pg in enumerate(Pg):
        trXg[i] = 0.5 * np.trace(np.dot(Kg[i], pg))
        #H = H * xg[i]
    X = suhf.integr_beta(trXg, fac='xg')
    return trXg, X

def get_S2(suhf, Pg_ortho):
    norb = int(Pg_ortho[0].shape[0]/2)
    S2g = np.zeros(len(Pg_ortho))
    for i,pg in enumerate(Pg_ortho):
        pgaa = pg[:norb, :norb] # ortho ao
        #print(pgaa)
        pgab = pg[:norb, norb:]
        pgba = pg[norb:, :norb]
        pgbb = pg[norb:, norb:]
        Pc = 0.5 * (pgaa + pgbb)
        Mz = 0.5 * (pgaa - pgbb)
        My = -0.5j * (pgab - pgba)
        #print(My)
        Mx = 0.5 * (pgab + pgba)
        #print(Pc)
        trPc, trMx, trMy, trMz = list(map(np.trace, [Pc, Mx, My, Mz]))
        if suhf.debug: print(trPc, trMx, trMy, trMz)
        Pc2 = np.dot(Pc, Pc)
        Mz2 = np.dot(Mz, Mz)
        My2 = np.dot(My, My).real
        Mx2 = np.dot(Mx, Mx)
        trPc2, trMx2, trMy2, trMz2 = list(map(np.trace, [Pc2, Mx2, My2, Mz2]))
        if suhf.debug: print(trPc2, trMx2, trMy2, trMz2)
        S2g[i] = trMx**2 + (trMy**2).real + trMz**2 + 0.5*(trMx2 + trMy2 + trMz2) + 1.5*(trPc - trPc2)
    S2 = suhf.integr_beta(S2g, fac='xg')
    print('S2 = %.6f'% S2)
    return S2

def get_Yg(suhf, Dg, Ng, dm_no, occ):
    norb = len(dm_no[0])
    vir = norb - occ
    Xg = []
    for i, ng in enumerate(Ng):
        Xgi1 = einsum('ij,jk,kl->il', Dg[i], dm_no[:,:occ], Ng[i]) 
        Xgi2 = einsum('ij,jk,kl->il', Ng[i], dm_no[:occ,:], Dg[i])
        #print(Xgi1)
        #print(Xgi2)
        Xgi = np.hstack((Xgi1, np.zeros((norb, vir)))) \
            + np.vstack((Xgi2, np.zeros((vir, norb)))) 
        #print(Xgi)
        Xg.append(Xgi)
    if suhf.debug:
        print('X(g)')
        print(Xg[0])
    Xg = np.array(Xg)
    Xg_int = suhf.integr_beta(Xg, fac='xg')
    if suhf.debug:
        print('X(g) int')
        print(Xg_int)
    Yg = Xg - Xg_int
    return Xg, Xg_int, Yg

def get_Feff(suhf, trHg, Gg, Ng, Pg, Dg, occ, Yg, Xg, F_ortho):
    hcore_no = suhf.hcore_no
    dm_ortho = suhf.dm_ortho
    dm_no = suhf.dm_no
    no = suhf.no
    norb = len(Pg[0])
    vir = norb - occ
    Feff_g = [] 
    Feff0 = []
    for i, pg in enumerate(Pg):
        feff0 = Yg[i] * trHg[i]
        #print(feff0)
        fg = hcore_no + Gg[i]
        feff1 = einsum('ij,jk,kl,lm,mn->in', Ng[i], dm_no[:occ,:], fg, np.eye(norb)-pg, Dg[i])
        #print(feff1)
        feff1 = np.vstack((feff1, np.zeros((vir, norb))))
        feff2 = einsum('ij,jk,kl,lm,mn->in', np.eye(norb)-pg, fg, Dg[i], dm_no[:, :occ], Ng[i])
        #print(feff2)
        feff2 = np.hstack((feff2, np.zeros((norb, vir))))
        feff = feff0 + feff1 + feff2
        Feff_g.append(feff)
        #Feff0.append(feff0)
        #print(feff)
    Feff_g = np.array(Feff_g)
    #Feff0 = suhf.integr_beta(np.array(Feff0), fac='xg')
    #print(Feff0)
    Feff = suhf.integr_beta(Feff_g, fac='xg')
    if suhf.debug2: print('Feff\n', Feff)
    Feff_ortho = einsum('ij,jk,lk->il', no, Feff, no)
    Feff_ortho = 0.5 * (Feff_ortho + Feff_ortho.T)
    if suhf.debug2: print('Feff (ortho)\n', Feff_ortho)

    lenf = len(F_ortho[0])
    F_ortho = np.array(F_ortho)
    Feff_ortho = np.array([Feff_ortho[:lenf, :lenf], Feff_ortho[lenf:,lenf:]])
    Q = np.eye(len(dm_ortho[0])) - dm_ortho
    Foo = einsum('tij,tjk,tkl->til', dm_ortho, F_ortho, dm_ortho)
    Fvv = einsum('tij,tjk,tkl->til', Q, F_ortho, Q)
    Fov = einsum('tij,tjk,tkl->til', dm_ortho, Feff_ortho, Q)
    Fvo = einsum('tij,tjk,tkl->til', Q, Feff_ortho, dm_ortho)
    #F_mod_ortho_a = np.vstack((np.hstack((Foo[0], Fov[0])), np.hstack((Fvo[0], Fvv[0]))))
    #F_mod_ortho_b = np.vstack((np.hstack((Foo[1], Fov[1])), np.hstack((Fvo[1], Fvv[1]))))
    F_mod_ortho_a = Foo[0] + Fov[0] + Fvo[0] + Fvv[0]
    F_mod_ortho_b = Foo[1] + Fov[1] + Fvo[1] + Fvv[1]
    F_mod_ortho_a = 0.5*(F_mod_ortho_a + F_mod_ortho_a.T)
    F_mod_ortho_b = 0.5*(F_mod_ortho_b + F_mod_ortho_b.T)
    F_mod_ortho = np.vstack((
        np.hstack((F_mod_ortho_a, np.zeros(F_mod_ortho_a.shape))),
        np.hstack((np.zeros(F_mod_ortho_b.shape), F_mod_ortho_b))
    ))
    if suhf.debug:
        print('Feff (mod,ortho)')
        print(F_mod_ortho)
    return Feff_ortho, F_mod_ortho

def Diag_Feff(F):
    e_a, v_a = np.linalg.eigh(F[0])
    e_b, v_b = np.linalg.eigh(F[1])
#    P_a = einsum('ij,kj->ik', v_a[:,:na], v_a[:,:na])
#    P_b = einsum('ij,kj->ik', v_b[:,:nb], v_b[:,:nb])
    return [e_a, e_b], [v_a, v_b] #, np.array([P_a, P_b])

def make_dm(mo_coeff, mo_occ):
    return scf.uhf.make_rdm1(mo_coeff, mo_occ)

class SUHF():
    '''
    Required input:
        guesshf: UHF object
    Options:
        verbose: 4,6,8
        conv_tol: 1e-7 for RMSD
        max_cycle: 70
        diis_on: 
        level_shift:
    Output:
        E_suhf:
        mo_reg:
        dm_reg: deformed density matrix
        suhf_dm: 1pdm
        natocc: SUHF natural orbital occupation number
        natorb: SUHF natural orbital (regular basis) 
    '''
    orbsym = None

    def __init__(self, guesshf=None):
        self.guesshf = guesshf
        self.nbeta = 8

        self.cut_no = False
        #self.use_no = True
        self.verbose = 4
        self.printmo = False
        #self.debug = False
        self.output = None # since 0.3.1, define output is important, it decides self.chkfile
        self.restart = False
        self.conv_tol = 1e-7 # For RMSD
        self.max_cycle = 70
        self.noiter = False
        self.diis_on = True
        self.diis_driver = 'def'
        self.diis_space = None
        self.diis_damp = 0.0
        self.diis_start_cyc = None
        self.diis_start_damp = 0
        self.level_shift = None
        self.shift_driver = 'def'
        self.incfock = False
        self.ifsel = 2

        self.dft = False
        self.xc = None
        self.symm = False
        self.makedm = True
        self.do2pdm = False
        self.tofch = False
        self.oldfch = None
        self.dumpchk = False
        self.chkfile0 = None
        self.chkfile = None

        self.setmom = None
        self.mom_reforb = None
        self.mom_start_cyc = 5

        self.built = False

        self.E_suhf = None

    def dump_flags(self):
        print('\n******** %s ********' % self.__class__)
        print('Date: %s' % time.ctime())
        import pyphf
        print('pyphf version %s' % pyphf.__version__)
        info = util2.repo_info(os.path.join(__file__, '..', '..'))
        print('pyphf path  %s' % info['path'])
        if 'git' in info:
            print(info['git'] )
        print('max_cycle: %d' % self.max_cycle)
        self.debug = False
        self.debug2 = False
        if self.verbose <= 4:
            print('verbose: %d                # normal' % self.verbose)
        elif self.verbose <= 6:
            self.debug = True
            print('verbose: %d                # debug' % self.verbose)
        else:
            self.debug = True
            self.debug2 = True
            print('verbose: %d                # debug2' % self.verbose)
        print('conv_tol: %g           # %g for RMSD(dm), %g for MaxD(dm), %g for dE' % (
            self.conv_tol, self.conv_tol, self.conv_tol*1e2, self.conv_tol*1e-2))
        if self.conv_tol > 1e-5:
            print(util2.warn("conv_tol too large"))

    def build(self):
        self.dump_flags()
        if self.output is None:
            self.output = str(os.getpid())
        if self.guesshf is not None:
            hf = self.guesshf
            self.mol = hf.mol
            self.stdout = self.guesshf.stdout
            if self.dumpchk:
                self.chkfile0 = self.output + '_ges.pchk'
                #print(hf.mo_energy)
                #print(hf.mo_coeff)
                #print(hf.mo_occ)
                chkfile.dump_scf(hf.mol, self.chkfile0, hf.e_tot, hf.mo_energy,
                                 hf.mo_coeff, hf.mo_occ)
                print('chkfile0: %s # the file store hf for guess' % self.chkfile0)
        elif self.chkfile0 is not None:
            chkfile1 = self.output + '_ges.pchk'
            os.system('cp %s %s' % (self.chkfile0, chkfile1))
            print('Load UHF from %s, new chkfile %s' % (self.chkfile0, chkfile1))
            mol = lib.chkfile.load_mol(chkfile1)
            hf = scf.UHF(mol)
            hf.chkfile = chkfile1
            hf.init_guess = 'chkfile'
            hf.kernel()
            self.mol = hf.mol
            self.guesshf = hf
            print('****** End of UHF ********')
        elif self.chkfile is not None:
            self.mol, suinfo = util2.load_chk(self.chkfile)
        else:
            guess = ''' 
            guesshf: a UHF object
            chkfile0: a PySCF chkfile storing a UHF result
            chkfile: SUHF chkfile
            '''
            raise AttributeError('You must provide one of below as a guess:' + guess)
        if self.dumpchk:
            self.chkfile = self.output + '_su.pchk'
            print('chkfile:  %s  # the file store suhf info' % self.chkfile)
        #self.chkfile2 = self.output + '_no.pchk'
        #print('chkfile2: %s # the file store suhf NO' % self.chkfile2)

        self.max_memory = hf.max_memory
        if self.diis_on:
            #assert issubclass(mf.DIIS, lib.diis.DIIS)
            if self.diis_space is None:
                self.diis_space = 8
            if self.diis_start_cyc is None:
                self.diis_start_cyc = 10
            self.diis_file = None
            #mf_diis.rollback = mf.diis_space_rollback
            if self.diis_driver == 'def':
                self.diis = scf.diis.CDIIS()
            elif self.diis_driver == 'rev1':
                self.diis = util2.CDIISrev1()
                #self.diis.damp = self.diis_damp
            elif self.diis_driver == 'rev2' or self.diis_driver == 'plain':
                self.diis = util2.CDIISrev2()
            self.diis.space = self.diis_space
            #self.diis.damp = self.diis_damp
            print('DIIS: %s' % self.diis.__class__)
            print('diis_start_cyc = %d' % self.diis_start_cyc)
            print('diis_space = %d' % self.diis_space)
            #print('diis_damp = %g' % self.diis_damp)
        if self.level_shift is not None:
            shift = self.level_shift
            print('level shift: %.3f a.u.' % shift)
        #if self.output is not None:
        #    os.system("echo '' > %s" % self.output)
        #    sys.stdout = open(self.output, 'a')
        S = scf.hf.get_ovlp(self.mol)
        self.ovlp = S
        Ca, Cb = self.guesshf.mo_coeff
        
        Su, Ss, Sv = scipy.linalg.svd(S, lapack_driver='gesvd')
        X = einsum('ij,j->ij', Su, Ss**(-0.5))
        self.X = X
        XS = np.dot(X.T,S)
        self.XS = XS
        if self.debug:
            print('C (reg)\n', Ca, '\n', Cb)
            print('S\n', S)
            print('SVD: S^(-1/2)\n', X)
            print('S^(1/2)\n', XS)
        
        Ca_ortho = np.dot(XS, Ca)
        Cb_ortho = np.dot(XS, Cb)
        self.mo_ortho = Ca_ortho, Cb_ortho
        self.dm_ortho = scf.uhf.make_rdm1(self.mo_ortho, self.guesshf.mo_occ)
        self.dm_reg = None
        if self.debug:
            print('C (ortho)\n', Ca_ortho, '\n', Cb_ortho)
            print('density matrix (ortho)')
            print(self.dm_ortho)
        self.norb = len(self.dm_ortho[0])

        #self.nbeta = 8
        self.grids, self.weights = wigner.get_beta(self.nbeta)
        print('grids: ', self.grids, '\nweights: ', self.weights)
        self.sinbeta = np.sin(self.grids)
        spin = self.mol.spin
        na, nb = self.guesshf.nelec
        self.nelec = na, nb
        sz = (na-nb)/2
        print('S = %.1f, Sz = %.1f' % (spin/2, sz))
        self.S, self.Sz = spin/2, sz
        self.d_expr, self.d_func, self.d = wigner.wigner(spin, sz, self.nbeta, self.grids)
        self.E_suhf = None
        self.energy_nuc = self.mol.energy_nuc()
        hcore = hf.get_hcore()
        self.hcore_reg = hcore
        self.hcore_ortho = einsum('ji,jk,kl->il', X, hcore, X)
        if self.debug:
            print('hcore (ortho)\n', self.hcore_ortho)
        #hf.direct_scf_tol = 1e-13
        #self.vhfopt = hf.init_direct_scf()
        self.vhfopt = {None: hf.init_direct_scf(self.mol)}

        if self.dft:
            if self.xc is None:
                if getattr(self.guesshf, 'xc', None) is None:
                    raise AttributeError('self.xc needs to be set')
                else:
                    self.xc = self.guesshf.xc
            #ni = numint.NumInt()
            #self._numint = ni
            self._ks, self.ksgrids = sudft.set_grids(self.mol)
            #self.xc = self.guesshf.xc
            self._ks.xc = self.xc
            self._numint = self._ks._numint
            ni = self._numint
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.xc, spin=self.mol.spin)
            self.hyb = hyb
            print('dft on')
            print(f'xc: {self.xc}, hyb: {hyb}')
            if omega > 1e-10:
                #raise NotImplementedError('Range Separation not Implemented')
                self.omega = omega
                self.alpha = alpha
                print(f'omega: {omega}, alpha: {alpha}')
                mol = self.mol
                with mol.with_range_coulomb(omega):
                    self.vhfopt[omega] = hf.init_direct_scf(mol)
            else:
                self.omega = None
        else:
            self.hyb = None
            self.omega = None
        mo_occ = self.get_occ()
        self.mo_occ = mo_occ
        self.mom = False
        if self.setmom is not None:
            self.mom = True
            if self.mom_reforb is None:
                raise AttributeError('You need to provide mom_reforb, i.e. MOs from a previous SUHF obj')
            aexci, bexci = self.setmom
            self.setocc = deltascf.set_occ(mo_occ, aexci, bexci)

        self.built = True

    def integr_beta(self, q, fac='normal'):
        if fac=='xg':
            return integr_beta(q, self.d, self.grids, self.weights, 'xg', self.xg, self.ciS)
        elif fac=='ci':
            return integr_beta(q, self.d, self.grids, self.weights, 'ci', None, self.ciS)
        else:
            return integr_beta(q, self.d, self.grids, self.weights)

    
    def kernel(self):
        t_start = time.time()
        if not self.built:
            self.build()
        prec = 6
        linew = 160
        supp = True
        orbmax = 32
        self.precise = False
        if self.debug:
            linew = 200
            orbmax = 60
            if self.precise:
                prec = 10
                supp = False
        np.set_printoptions(precision=prec, linewidth=linew, suppress=supp, threshold = orbmax**2)
        X = self.X
        na, nb = self.nelec
        norb = self.norb
        #mf = self.guesshf
        mo_occ = self.mo_occ

        thresh = self.conv_tol
        max_cycle = self.max_cycle
        noiter = self.noiter
        cyc = 1
        conv = False
        self.conv = conv
        
        Pgao = None
        #old_Pgao = None
        #old_Ggao = None
        F_mod_ortho = None
        #print(vhfopt)
        t_pre = time.time() 
        print('time for Preparation before cyc: %.3f' % (t_pre-t_start))
        while(not conv):
            print('**** Start Cycle %d ****' % cyc)
            old_suhf = self.E_suhf
            old_dm = self.dm_ortho
            if Pgao is not None and self.incfock:
                old_Pgao = Pgao
                old_Ggao = Ggao
            else:
                old_Pgao = None
                old_Ggao = None
            #if cyc==0:
            #    veff = mf.get_veff(dm = dm)
            #else:
            t01 = time.time()
            dm_reg = einsum('ij,tjk,lk->til', X, self.dm_ortho, X)
            #veff = scf.uhf.get_veff(self.mol, dm_reg, vhfopt=self.vhfopt)
            veff = self.get_uhf_veff(dm_reg)
            veff_ortho = einsum('ji,tjk,kl->til', X, veff, X)
            if self.debug:
                print('dm (ortho)')
                print(self.dm_ortho)
            #Fa, Fb = hcore + veff
            #Fa_ortho = einsum('ji,jk,kl->il', X, Fa, X)
            #Fb_ortho = einsum('ji,jk,kl->il', X, Fb, X)
            Fa_ortho, Fb_ortho = self.hcore_ortho + veff_ortho
            F_ortho = Fa_ortho, Fb_ortho
            if self.debug:
                print('Fock (ortho)\n', F_ortho)
                #e_uhf, e_uhf_coul = scf.uhf.energy_elec(self.guesshf, self.dm_ortho, self.hcore_ortho, veff_ortho)
                #print('E(UHF) = %12.6f' % e_uhf)
#            if self.use_no:
            dm_no, dm_expanded, no = find_NO(self, self.dm_ortho, mo_occ)
            self.dm_no = dm_no
            self.no = no
            Dg, Ng, Pg = get_Ng(self.grids, self.no, self.dm_no, na+nb)
            self.Pg = Pg
            if self.debug:
                print('D(g) (NO)\n', Dg[0])
                print('N(g) (NO)\n', Ng[0])
                print('P(g) (NO)\n', Pg[0])
            t05 = time.time()
            print('time for NO, Ng: %.3f' % (t05-t01))
            #Gg, Gg_ortho, Pg_ortho, Pgao, Ggao = jk.get_Gg(self.mol, Pg, self.no, X, dm_last=old_Pgao, Ggao_last=old_Ggao, opt=self.vhfopt)
            Gg, Gg_ortho, Pg_ortho, Pgao, Ggao = self.get_Gg(dm_last=old_Pgao, Ggao_last=old_Ggao)
            self.Gg = Gg
            self.Gg_ortho = Gg_ortho
            if self.debug:
                print('Pg_ortho\n', Pg_ortho[0])
                print('G(g) (NO)\n' , Gg[0])
            t06 = time.time()
            print('time for Gg: %.3f' % (t06-t05))
            xg, yg, ciS, C_no = get_xg(self, self.no, mo_occ, Ng)
            self.xg, self.ciS = xg, ciS
            #yg, ciS = util.get_yg(self, xg)
            trHg, ciH, H_suhf = get_H(self, self.hcore_ortho, self.no, Pg, Gg, xg)
            self.ciH = ciH
            S2 = get_S2(self, Pg_ortho)
            Xg, Xg_int, Yg = get_Yg(self, Dg, Ng, self.dm_no, na+nb)
            F_last = F_mod_ortho
            Feff_ortho,  F_mod_ortho0 = get_Feff(self, trHg, Gg, Ng, Pg, Dg, na+nb, Yg, Xg, F_ortho)
            E_suhf = self.energy_nuc + H_suhf
            #print('E(SUHF) = %15.8f' % E_suhf)
            if self.ifsel == 1:
                F_mod_ortho = Feff_ortho
            elif self.ifsel == 2:
                Faa = F_mod_ortho0[:norb, :norb]
                Fbb = F_mod_ortho0[norb:, norb:]
                F_mod_ortho = np.array([Faa,Fbb])
            if self.dft:
                exc, vxc = self.ddft()
                E_suhf += exc
                # dft for noiter only, Fock is not well defined
                vxc_ortho = einsum('ji,tjk,kl->til', X, vxc, X)
                F_mod_ortho = F_mod_ortho + vxc_ortho

            self.E_suhf = E_suhf
            if old_suhf is not None:
                dE = E_suhf - old_suhf
                energy_rise = dE > 1e-7
            else:
                energy_rise = False
            if self.diis_on and cyc >= self.diis_start_cyc:
                s1e = np.eye(norb)
                errvec = scf.diis.get_err_vec(s1e, self.dm_ortho, F_mod_ortho, None)
                print('diis-norm(errvec)=%.6g'% np.linalg.norm(errvec))
                if cyc >= self.diis_start_damp and energy_rise:
                    self.diis.damp = self.diis_damp
                else:
                    self.diis.damp = 0.0
                print('diis_damp = %g' % self.diis.damp)
                F_mod_ortho = self.diis.update(s1e, self.dm_ortho, F_mod_ortho, f_prev=F_last)
                print('F(mod,ortho) updated with CDIIS')
                if self.debug: print(F_mod_ortho)
            if self.level_shift is not None:
                shift = self.level_shift
                s1e = np.eye(norb)
                #print('level shift: %.3f a.u.' % shift)
                if self.shift_driver == 'def':
                    F_mod_ortho = lev_shift(s1e, self.dm_ortho, F_mod_ortho, shift)
                elif self.shift_driver == '2':
                    F_mod_ortho = lev_shift2(s1e, self.dm_ortho, F_mod_ortho, shift)
            if noiter:
                self.regular()
                print(' E = %15.8f' % E_suhf)
                break  
            mo_e, mo_ortho = self.Diag_Feff(F_mod_ortho)
            mo_ortho = np.array(mo_ortho)
            #print(mo_ortho)
            dm_ortho = make_dm(mo_ortho, mo_occ)
            if self.debug or self.printmo:
                #print('e_a, e_b\n', mo_e[0], '\n', mo_e[1])
                print('v_a, v_b\n', mo_ortho[0], '\n', mo_ortho[1])
                print('P_a, P_b\n', dm_ortho[0],'\n', dm_ortho[1])
            self.dm_ortho = dm_ortho
            self.mo_ortho = mo_ortho
            self.mo_e = mo_e
            self.regular()
            if self.mom and cyc >= self.mom_start_cyc:
                mo_occ = deltascf.mom_occ(self, self.mom_reforb, self.setocc)
            else:
                mo_occ = self.get_occ(mo_e, mo_ortho)
            self.mo_occ = mo_occ
            self.dump_moe(mo_e, na, nb, mo_occ=mo_occ, orbsym=self.orbsym)
            t10 = time.time()
            print('time for xg, H, S2, Yg, Feff: %.3f' % (t10-t06))
        
            if old_suhf is not None:
                dE = E_suhf - old_suhf
                ddm = dm_ortho - old_dm
                conv = conv_check(E_suhf, dE, ddm, thresh, cyc)
            else:
                print(' E = %15.8f' % E_suhf)
            self.conv = conv
            cyc += 1
            if cyc >= max_cycle:
                print(util2.warn('SUHF not converged'))
                break

        # extra cycle to remove level shift
        old_suhf = self.E_suhf
        old_dm = self.dm_ortho
        old_Pgao = None
        old_Ggao = None
        if self.level_shift is not None and not noiter:
            print('**** Extra Cycle %d ****' % cyc)
            #veff = scf.uhf.get_veff(self.mol, dm_reg, vhfopt=self.vhfopt)
            dm_reg = self.dm_reg
            veff = self.get_uhf_veff(dm_reg)
            veff_ortho = einsum('ji,tjk,kl->til', X, veff, X)
            if self.debug:
                print('dm (ortho)')
                print(self.dm_ortho)
            Fa_ortho, Fb_ortho = self.hcore_ortho + veff_ortho
            F_ortho = Fa_ortho, Fb_ortho
            if self.debug:
                print('Fock (ortho)\n', F_ortho)
                #e_uhf, e_uhf_coul = scf.uhf.energy_elec(mf, self.dm_ortho, self.hcore_ortho, veff_ortho)
                #print('E(UHF) = %12.6f' % e_uhf)
            dm_no, dm_expanded, no = find_NO(self, self.dm_ortho, mo_occ)
            self.dm_no = dm_no
            self.no = no
            Dg, Ng, Pg = get_Ng(self.grids, self.no, self.dm_no, na+nb)
            self.Pg = Pg
            if self.debug:
                print('D(g) (NO)\n', Dg[0])
                print('N(g) (NO)\n', Ng[0])
                print('P(g) (NO)\n', Pg[0])
            #Gg, Gg_ortho, Pg_ortho, _, _ = jk.get_Gg(self.mol, Pg, self.no, X, opt=self.vhfopt)
            Gg, Gg_ortho, Pg_ortho, _, _ = self.get_Gg()
            self.Gg = Gg
            self.Gg_ortho = Gg_ortho
            if self.debug:
                print('Pg_ortho\n', Pg_ortho[0])
                print('G(g) (NO)\n' , Gg[0])
            xg, yg, ciS, C_no = get_xg(self, self.no, mo_occ, Ng)
            self.xg, self.ciS = xg, ciS
            #yg, ciS = util.get_yg(self, xg)
            trHg, ciH, H_suhf = get_H(self, self.hcore_ortho, self.no, Pg, Gg, xg)
            self.ciH = ciH
            S2 = get_S2(self, Pg_ortho)
            Xg, Xg_int, Yg = get_Yg(self, Dg, Ng, self.dm_no, na+nb)
            Feff_ortho, F_mod_ortho = get_Feff(self, trHg, Gg, Ng, Pg, Dg, na+nb, Yg, Xg, F_ortho)
            E_suhf = self.energy_nuc + H_suhf
            if self.ifsel == 1:
                F_mod_ortho = Feff_ortho
            elif self.ifsel == 2:
                Faa = F_mod_ortho0[:norb, :norb]
                Fbb = F_mod_ortho0[norb:, norb:]
                F_mod_ortho = np.array([Faa,Fbb])
            if self.dft:
                exc, vxc = self.ddft()
                E_suhf += exc
                # dft for noiter only, Fock is not well defined
                vxc_ortho = einsum('ji,tjk,kl->til', X, vxc, X)
                F_mod_ortho = F_mod_ortho + vxc_ortho
            
            self.E_suhf = E_suhf
            mo_e, mo_ortho = self.Diag_Feff(F_mod_ortho)
            mo_ortho = np.array(mo_ortho)
            dm_ortho = make_dm(mo_ortho, mo_occ)
            if self.debug or self.printmo:
                #print('e_a, e_b\n', mo_e[0], '\n', mo_e[1])
                print('v_a, v_b\n', mo_ortho[0], '\n', mo_ortho[1])
                print('P_a, P_b\n', dm_ortho[0],'\n', dm_ortho[1])
            self.dm_ortho = dm_ortho
            self.mo_ortho = mo_ortho
            self.mo_e = mo_e
            self.regular()
            if self.mom and cyc >= self.mom_start_cyc:
                mo_occ = deltascf.mom_occ(self, self.mom_reforb, self.setocc)
            else:
                mo_occ = self.get_occ(mo_e, mo_ortho)
            self.mo_occ = mo_occ
            self.dump_moe(mo_e, na, nb, mo_occ=mo_occ, orbsym=self.orbsym)

            #if old_suhf is not None:
            dE = E_suhf - old_suhf
            ddm = dm_ortho - old_dm
            max_ddm = abs(ddm).max()
            norm_ddm = np.linalg.norm(ddm) / (ddm.shape[-1]*np.sqrt(2))
            print('Extra E(SUHF) = %15.8f, delta E = %10.6g, MaxD(dm) = %10.6g, RMSD(dm) = %10.6g' % (E_suhf, dE, max_ddm, norm_ddm))
        t_aftercyc = time.time()
        print('time for cyc: %.3f' % (t_aftercyc-t_pre))

        if self.dumpchk:
            util2.dump_chk(self.mol, self.chkfile, self.E_suhf, self.mo_e, self.mo_reg, self.mo_occ, self.dm_reg)
        if self.makedm:
            suhf_dm = sudm.make_1pdm(self, Dg, self.dm_no, C_no)
            t_dm = time.time()
            print('time for 1pdm (and 2pdm): %.3f' % (t_dm-t_aftercyc))
            self.suhf_dm = suhf_dm
            self.natorb, self.natocc = sudm.natorb(self, suhf_dm)
            if self.tofch:
                S = self.mol.intor_symmetric('int1e_ovlp')
                util2.tofch(self.oldfch, self.natorb[2], self.natocc[2], S)
                #util2.tofch(self.oldfch, mo_reg, self.mo_e, S, 'SUHFMO')
            t_nat = time.time()
            print('time for natorb: %.3f' % (t_nat-t_dm))
            ss, s = scf.uhf.spin_square((self.mo_reg[0][:,self.mo_occ[0]>0],
                                self.mo_reg[1][:,self.mo_occ[1]>0]), self.ovlp)
            self.fake_s = s
            print('deformed <S^2> = %.8g  2S+1 = %.8g' % (ss, s))

        t_end = time.time()
        print('***** End of SUHF *****')
        print('time tot: %.3f' % (t_end-t_start))
        print('Date: %s' % time.ctime())
        return E_suhf, self.conv
    
    get_occ = get_occ

    dump_moe = util2.dump_moe

    def Diag_Feff(self, F):
        return Diag_Feff(F)

    def density_fit(self, auxbasis=None):
        #with_df = jk.DF(self.mol)
        with_df = self.guesshf.with_df
        #with_df.max_memory = self.max_memory
        #with_df.verbose = self.verbose
        #with_df.auxbasis = auxbasis
        dfmf = DFSUHF(self)
        dfmf.with_df = with_df
        return lib.set_class(dfmf, (DFSUHF, self.__class__))

    def get_uhf_veff(self, dm_reg):
        if self.dft:
            hyb = self.hyb
            #veff = self._ks.get_veff(dm=dm_reg)
            vj, vk = scf.hf.get_jk(self.mol, dm_reg, vhfopt=self.vhfopt[None])
            vk = vk * hyb
            if self.omega is not None:
                omega=self.omega
                alpha=self.alpha
                vklr = scf.hf.get_jk(self.mol, dm_reg, with_j=False, vhfopt=self.vhfopt[omega])[1]
                vk = vk + vklr * (alpha-hyb) 
            veff = vj[0] + vj[1] - vk
        else:
            veff = scf.uhf.get_veff(self.mol, dm_reg, vhfopt=self.vhfopt[None])
        return veff

    def get_Gg(self, dm_last=None, Ggao_last=None):
        if self.omega is not None:
            rsh = self.omega, self.alpha, self.hyb
        else:
            rsh = None
        return jk.get_Gg(self.mol, self.Pg, self.no, self.X, dm_last=dm_last, Ggao_last=Ggao_last, opt=self.vhfopt, 
                         hyb=self.hyb, rsh=rsh)

    def get_JKg(self, hyb=None):
        if self.omega is not None:
            raise NotImplementedError()
        return jk.get_JKg(self.mol, self.Pg, self.no, self.X, hyb=hyb)[:2]

    def get_EJK(self):
        Jg, Kg = self.get_JKg()
        EJ = get_EX(self, self.no, self.Pg, Jg, self.xg)[1]
        EK = get_EX(self, self.no, self.Pg, Kg, self.xg)[1]
        Jg, Kg = self.get_JKg(hyb=self.hyb)
        EK1 = get_EX(self, self.no, self.Pg, Kg, self.xg)[1]
        print('EJ = %.8f, EK = %.8f, EK1 = %.8f' % (EJ, EK, EK1))
        return EJ, EK
    
    def get_E1(self):
        E1 = get_E1(self, self.hcore_no, self.Pg)
        return E1

    def decomp(self):
        self.get_EJK()
        self.get_E1()
    
    fchk = util2.fchk
    
    def ddft(self, xc=None):
        if self.dm_reg is None:
            X = self.X
            self.dm_reg = einsum('ij,tjk,lk->til', X, self.dm_ortho, X) # regular ao
        if xc is None:
            xc = self.xc
        #ni = numint.NumInt()
        ni = self._numint
        if self.ksgrids is None:
            _, self.ksgrids = sudft.set_grids(self.mol)
        n, exc, vxc = ni.nr_uks(self.mol, self.ksgrids, xc, self.dm_reg)
        #omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xc, spin=self.mol.spin)
        #if omega > 1e-10: raise NotImplementedError('Range Separation not Implemented')
        #if hyb > 1e-10:
        #    ex_hf = self.get_EX()
        #    exc -= (1-hyb)*ex_hf
        #print('e_dft-e_hf(%s): %.6f ' % (xc,exc))
        return exc, vxc

    def to_hf(self):
        self.dft = False
        self.hyb = None
        self.omega = None
        return self
    
    def mo_reg2ortho(self, mo):
        if mo.ndim == 3:
            mo_ortho = einsum('ij,tjk->tik', self.XS, mo)
        elif mo.ndim == 2:
            mo_ortho = einsum('ij,jk->ik', self.XS, mo)
        else:
            raise ValueError('mo should be 2D or 3D array')
        return mo_ortho
    
    def mo_ortho2reg(self, mo):
        if mo.ndim == 3:
            mo_reg = einsum('ij,tjk->tik', self.X, mo)
        elif mo.ndim == 2:
            mo_reg = einsum('ij,jk->ik', self.X, mo)
        else:
            raise ValueError('mo should be 2D or 3D array')
        return mo_reg
    
    def f_ortho2reg(self, f):
        return einsum('ji,tjk,kl->til', self.XS, f, self.XS)

    def regular(self):
        X = self.X
        dm_reg = einsum('ij,tjk,lk->til', X, self.dm_ortho, X) # regular ao
        #mo_ortho = np.array(self.mo_ortho)
        mo_reg = einsum('ij,tjk->tik', X, self.mo_ortho)
        self.dm_reg = dm_reg
        self.mo_reg = mo_reg
        if self.debug or self.printmo:
            print('dm_reg\n', dm_reg)
            print('mo_reg\n', mo_reg[0], '\n', mo_reg[1])

    def get_Fg(self):
        X = self.X
        Fg_ortho = []
        Fg_reg = []
        for g in self.Gg_ortho:
            f_ortho = self.hcore_ortho + g
            f_reg = einsum('ij,tjk,lk->til', X, f_ortho)
            Fg_ortho.append(f_ortho)
            Fg_reg.append(f_reg)
        return Fg_ortho, Fg_reg
        
def conv_check(E_suhf, dE, ddm, thresh, cyc):
    max_ddm = abs(ddm).max()
    norm_ddm = np.linalg.norm(ddm) / (ddm.shape[-1]*np.sqrt(2))
    if abs(dE)< (thresh*1e-2) and max_ddm < (thresh*1e2) and norm_ddm < thresh:
        conv = True
        print('\n***************')
        print('SUHF converged at cycle %d' %cyc)
        print('Final E(SUHF) = %15.8f, delta E = %10.6g, MaxD(dm) = %10.6g, RMSD(dm) = %10.6g' % (E_suhf, dE, max_ddm, norm_ddm))
    else:
        conv = False
        print(' E = %15.8f, delta E = %10.6g, MaxD(dm) = %10.6g, RMSD(dm) = %10.6g' % (E_suhf, dE, max_ddm, norm_ddm))
    return conv

def lev_shift(s, dm, f, shift):
    new_f = (scf.hf.level_shift(s, dm[0], f[0], shift),
             scf.hf.level_shift(s, dm[1], f[1], shift)
             )
    return np.array(new_f)

def lev_shift2(s, dm, f, shift):
    new_f = (shift_down_occ(s, dm[0], f[0], shift),
             shift_down_occ(s, dm[1], f[1], shift)
             )
    return np.array(new_f)

def shift_down_occ(s, dm, f, shift):
    # assume ortho
    f = f - shift * dm
    return f

class DFSUHF(SUHF):

    __name_mixin__ = 'DF'

    def __init__(self, smf):
        self.__dict__.update(smf.__dict__)

    def get_Gg(self, dm_last=None, Ggao_last=None):
        if self.omega is not None:
            rsh = self.omega, self.alpha, self.hyb
        else:
            rsh = None
        return jk.get_Gg_df(self.mol, self.Pg, self.no, self.X, #dm_last=dm_last, Ggao_last=Ggao_last, opt=self.vhfopt
                            with_df=self.with_df, rsh=rsh)

    def get_uhf_veff(self, dm_reg):
        if self.dft:
            hyb = self.hyb
            dfobj = self.with_df
            #veff = self._ks.get_veff(dm=dm_reg)
            vj, vk = jk.get_jk_df(dfobj, dm_reg, hermi=0)
            vk = vk * hyb
            if self.omega is not None:
                omega=self.omega
                alpha=self.alpha
                with dfobj.range_coulomb(omega) as rsh_df:
                    vklr = jk.get_jk_df(rsh_df, dm_reg, with_j=False, hermi=0)[1]
                vk = vk + vklr * (alpha-hyb) 
        else:
            vj, vk = jk.get_jk_df(self.with_df, dm_reg, hermi=0)
        veff = vj[0] + vj[1] - vk
        return veff
