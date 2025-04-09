from pyphf import util2, sudm
from pyscf import dft
import pyscf.dft.numint as numint
#from pyphf import numint

import numpy as np
from functools import partial
import time

print = partial(print, flush=True)
einsum = partial(np.einsum, optimize=True)

'''
SUHF+DFT, SUHF+fDFT, SUHF+fcDFT
 Garza, A. J.; Jiménez-Hoyos, C. A.; Scuseria, G. E. J Chem Phys 2013, 138, 134102
 Garza, A. J.; Jiménez-Hoyos, C. A.; Scuseria, G. E. J Chem Phys 2014, 140, 244102
'''

class SUDFT():
    def __init__(self, suhf):
        #suhf = util.SUHF(guesshf)
        print('\n******** %s ********' % self.__class__)
        self.suhf = suhf
        self.suxc = 'tpss'
        self.grids = 'default'
        self.output = None
        self.dens = 'deformed' # or relaxed
        self.trunc = None
        self.nref = None
        self.ncore = None

    def kernel(self):
        if self.suhf.E_suhf is None:
            self.suhf.kernel()
        print('density: %s' % self.dens)
        print('truncation: %s' % self.trunc)
        #if self.output is not None:
        #    sys.stdout = open(self.output, 'a')
        print('***** Start DFT Correlation for SUHF+DFT **********')
        t1 = time.time()
        mol = self.suhf.mol
        E_suhf = self.suhf.E_suhf
        #dm_ortho = self.suhf.dm_ortho
        #X = self.suhf.X
        #dm_reg = np.einsum('ij,tjk,lk->til', X, dm_ortho, X)
        if self.dens[:3] == 'def':
            dm = self.suhf.dm_reg
        elif self.dens[:3] == 'rel':
            dm = self.suhf.suhf_dm
        gamma = self.suhf.suhf_dm
        gamma = gamma[0] + gamma[1]
        
        grids = set_grids(mol, self.grids)
        ni = numint.NumInt()
        if self.trunc is None:
            if self.suxc == 'CS':
                suxc = 'MGGA_C_CS'
                n, exc = get_exc(ni, mol, grids, 'HF,%s'%suxc, dm)
            elif self.suxc.upper() == 'TPSS_MOD':
                suxc = 'TPSS'
                n, exc = get_exc(ni, mol, grids, 'HF,%s'%suxc, dm, special=1)
            else:
                n, exc, vxc = ni.nr_uks(mol, grids, 'HF,%s'%self.suxc, dm)
        elif self.trunc == 'f' or self.trunc == 'fc':
            natorb = self.suhf.natorb[2]
            natocc = self.suhf.natocc[2]
            #natocc = natocc[0] + natocc[1]
            #print('natocc', natocc)
            if self.nref is None:
                ref = [2.0 if occ > 1e-2 else 0.0 for occ in natocc]
            else:
                nref = self.nref
                nmo = len(natocc)
                ref = [2.0 if i < nref else 0.0 for i in range(nmo)]
            refdump, [refc, refa, refe] = util2.dump_occ(ref, 2.0)
            print('ref: ', refdump)
            ref = np.array(ref)
            dm_ref = einsum('ij,j,kj -> ik', natorb, ref, natorb)
            special=0
            if self.suxc == 'CS':
                suxc = 'MGGA_C_CS'
            elif self.suxc.upper() == 'TPSS_MOD':
                suxc = 'TPSS'
                special=1
            else:
                suxc = self.suxc
            n, exc, excf = get_exc(ni, mol, grids, 'HF,%s'%suxc, dm, trunc='f', gamma=gamma, dmref=dm_ref, special=special)
        elif self.trunc == 'fd':
            # +fDFT with exact Garza style 
            defm_no, defm_occ = sudm.natorb(self.suhf, dm)
            no = defm_no[2]
            nocc = defm_occ[2]
            nref = self.nref
            nmo = len(nocc)
            #ref = [2.0 if occ > 1e-2 else 0.0 for occ in nocc]
            ref = [2.0 if i < nref else 0.0 for i in range(nmo)]
            refdump, [refc, refa, refe] = util2.dump_occ(ref, 2.0)
            print('ref: ', refdump)
            ref = np.array(ref)
            dm_ref = einsum('ij,j,kj -> ik', no, ref, no)
            n, exc, excf = get_exc(ni, mol, grids, 'HF,%s'% self.suxc, dm, trunc='f', dmref=dm_ref)


        if self.trunc == 'fc':
            if self.ncore is None:
                core = [2.0 if occ > 1.99 else 0.0 for occ in natocc]
            else:
                nmo = len(natocc)
                core = [2.0 if i < self.ncore else 0.0 for i in range(nmo)]
            print('core', core)
            core = np.array(core)
            dm_core = einsum('ij,j,kj -> ik', natorb, core, natorb)
            #n1, exc1, core1 = get_exc(ni, self.suhf.mol, ks.grids, 'HF,%s'%suxc, dm_core, trunc='f', dmref=dm)
            n2, exc2, dE_fc = get_exc(ni, mol, grids, 'HF,%s'%suxc, dm_core, trunc='fc', gamma=gamma, dmref=dm_ref, dmcore=dm_core)
            #print('core1 %f, core2 %f' % (core1, core2))
            #dE_fc = core1 - core2
            print('dE_fc: %f' % dE_fc)
            excfc = excf + dE_fc

        E_sudft = E_suhf + exc
        print("E(SUHF) = %15.8f" % E_suhf)
        print("E_c(%s) = %15.8f" % (self.suxc.upper(), exc))
        print("E(SUHF+DFT) = %15.8f" % E_sudft)
        if self.trunc[0] == 'f':
            E_sufdft = E_suhf + excf
            print("f E_c(%s) = %15.8f" % (self.suxc.upper(), excf))
            print("E(SUHF+fDFT) = %15.8f" % E_sufdft)
        if self.trunc == 'fc':
            E_sufcdft = E_suhf + excfc
            print("fc E_c(%s) = %15.8f" % (self.suxc.upper(), excfc))
            print("E(SUHF+fcDFT) = %15.8f" % E_sufcdft)
        
        t2 = time.time()
        print('time for DFT: %.3f' % (t2-t1))
        return exc, E_sudft

def set_grids(mol, grids='def', debug=False):
    ks = dft.UKS(mol)
    if grids[:5] == 'ultra':
        ks.grids.atom_grid = (99, 590)
    elif grids[:4] == 'fine':
        ks.grids.atom_grid = (75, 302)
    #ks.grids.verbose = 1
    ks.grids.build()
    if debug:
        print('grids: ', ks.grids.atom_grid, '\n', ks.grids.coords.shape)
    return ks, ks.grids

def get_exc(ni, mol, grids, xc_code, dms, trunc=None, gamma=None, dmref=None, dmcore=None,
            relativity=0, hermi=0, max_memory=2000, verbose=9, special=0):
    '''
    modified from pyscf.dft.numint.nr_uks
    special = 1 for TPSS modified
    '''
    xctype = ni._xc_type(xc_code)
    #if xctype == 'NLC':
    #    dms_sf = dms[0] + dms[1]
    #    nelec, excsum, vmat = nr_rks(ni, mol, grids, xc_code, dms_sf, relativity, hermi,
    #                                 max_memory, verbose)
    #    return [nelec,nelec], excsum, np.asarray([vmat,vmat])

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    dma, dmb = numint._format_uks_dm(dms)
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dma, hermi)[:2]
    make_rhob       = ni._gen_rho_evaluator(mol, dmb, hermi)[0]
    make_rho_ref    = ni._gen_rho_evaluator(mol, dmref, hermi)[0]
    make_gamma      = ni._gen_rho_evaluator(mol, gamma, hermi)[0]
    if trunc == 'fc':
        make_rho_core       = ni._gen_rho_evaluator(mol, dmcore, hermi)[0]

    nelec = np.zeros((2,nset))
    nelec_g = np.zeros(nset)
    excsum = np.zeros(nset)
    excfsum = np.zeros(nset)
#    vmat = np.zeros((2,nset,nao,nao), dtype=np.result_type(dma, dmb))
#    aow = None
    if xctype == 'LDA':
        ao_deriv = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
#            aow = np.ndarray(ao.shape, order='F', buffer=aow)
            for idm in range(nset):
                rho_a = make_rhoa(idm, ao, mask, xctype)
                rho_b = make_rhob(idm, ao, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b), spin=1,
                                      relativity=relativity, deriv=1,
                                      verbose=verbose)[:2]
                vrho = vxc[0]
                den = rho_a * weight
                nelec[0,idm] += den.sum()
                excsum[idm] += np.dot(den, exc)
                den = rho_b * weight
                nelec[1,idm] += den.sum()
                excsum[idm] += np.dot(den, exc)

#                # *.5 due to +c.c. in the end
#                #:aow = np.einsum('pi,p->pi', ao, .5*weight*vrho[:,0], out=aow)
#                aow = _scale_ao(ao, .5*weight*vrho[:,0], out=aow)
#                vmat[0,idm] += _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
#                #:aow = np.einsum('pi,p->pi', ao, .5*weight*vrho[:,1], out=aow)
#                aow = _scale_ao(ao, .5*weight*vrho[:,1], out=aow)
#                vmat[1,idm] += _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
                rho_a = rho_b = exc = vxc = vrho = None
    elif xctype == 'GGA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
#            aow = np.ndarray(ao[0].shape, order='F', buffer=aow)
            for idm in range(nset):
                rho_a = make_rhoa(idm, ao, mask, xctype)
                rho_b = make_rhob(idm, ao, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b), spin=1,
                                      relativity=relativity, deriv=1,
                                      verbose=verbose)[:2]
                den = rho_a[0]*weight
                nelec[0,idm] += den.sum()
                excsum[idm] += np.dot(den, exc)
                den = rho_b[0]*weight
                nelec[1,idm] += den.sum()
                excsum[idm] += np.dot(den, exc)

#                wva, wvb = _uks_gga_wv0((rho_a,rho_b), vxc, weight)
#                #:aow = np.einsum('npi,np->pi', ao, wva, out=aow)
#                aow = _scale_ao(ao, wva, out=aow)
#                vmat[0,idm] += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
#                #:aow = np.einsum('npi,np->pi', ao, wvb, out=aow)
#                aow = _scale_ao(ao, wvb, out=aow)
#                vmat[1,idm] += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
                rho_a = rho_b = exc = vxc = wva = wvb = None
    elif xctype == 'MGGA':
#        if (any(x in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00'))):
#            raise NotImplementedError('laplacian in meta-GGA method')
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
#            aow = np.ndarray(ao[0].shape, order='F', buffer=aow)
            for idm in range(nset):
                rho_a = make_rhoa(idm, ao, mask, xctype)
                rho_b = make_rhob(idm, ao, mask, xctype)
                rho_ref = make_rho_ref(idm, ao, mask, xctype)
                gamma = make_gamma(idm, ao, mask, xctype)
                if trunc=='fc':
                    rho_core = make_rho_core(idm, ao, mask, xctype)
                exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b), spin=1,
                                      relativity=relativity, deriv=1,
                                      verbose=verbose)[:2]
                #exc, vxc = ni.eval_xc(xc_code, (rho_a, rho_b), spin=1,
                #                      relativity=relativity, deriv=1,
                #                      verbose=verbose, special=special)[:2]
                vrho, vsigma, vlapl, vtau = vxc[:4]
                den_a = rho_a[0]*weight
                nelec[0,idm] += den_a.sum()
                excsum[idm] += np.dot(den_a, exc)

                den_b = rho_b[0]*weight
                nelec[1,idm] += den_b.sum()
                excsum[idm] += np.dot(den_b, exc)
                rhoall = rho_a[0] + rho_b[0]
                #print('f', f.shape)
                #if trunc == 'f':
                eta = get_eta(rho_ref[0], gamma[0])
                print('rhoall', rhoall.min(), rhoall.max())
                print('gamma', gamma[0].min(), gamma[0].max())
                nelec_g[idm] += (gamma[0]*weight).sum()
                print('eta', eta.min(), eta.max())
                if trunc == 'fc':
                    eta1 = get_eta(gamma[0], rho_core[0])
                    print('eta1', eta1.min(), eta1.max())
                    eta2 = get_eta(rho_ref[0], rho_core[0])
                    print('eta2', eta2.min(), eta2.max())
                #if trunc == 'f':
                f = get_f(gamma[0], eta)
                if trunc == 'fc':
                    f = get_f(rho_core[0], eta1) - get_f(rho_core[0], eta2)
                excfsum[idm] += einsum('i,i,i->', den_a+den_b, exc, f)
                print('f', f.min(), f.max())

#                wva, wvb = _uks_gga_wv0((rho_a,rho_b), vxc, weight)
#                #:aow = np.einsum('npi,np->pi', ao[:4], wva, out=aow)
#                aow = _scale_ao(ao[:4], wva, out=aow)
#                vmat[0,idm] += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
#                #:aow = np.einsum('npi,np->pi', ao[:4], wvb, out=aow)
#                aow = _scale_ao(ao[:4], wvb, out=aow)
#                vmat[1,idm] += _dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)

# FIXME: .5 * .5   First 0.5 for v+v.T symmetrization.
# Second 0.5 is due to the Libxc convention tau = 1/2 \nabla\phi\dot\nabla\phi
#                wv = (.25 * weight * vtau[:,0]).reshape(-1,1)
#                vmat[0,idm] += _dot_ao_ao(mol, ao[1], wv*ao[1], mask, shls_slice, ao_loc)
#                vmat[0,idm] += _dot_ao_ao(mol, ao[2], wv*ao[2], mask, shls_slice, ao_loc)
#                vmat[0,idm] += _dot_ao_ao(mol, ao[3], wv*ao[3], mask, shls_slice, ao_loc)
#                wv = (.25 * weight * vtau[:,1]).reshape(-1,1)
#                vmat[1,idm] += _dot_ao_ao(mol, ao[1], wv*ao[1], mask, shls_slice, ao_loc)
#                vmat[1,idm] += _dot_ao_ao(mol, ao[2], wv*ao[2], mask, shls_slice, ao_loc)
#                vmat[1,idm] += _dot_ao_ao(mol, ao[3], wv*ao[3], mask, shls_slice, ao_loc)
                rho_a = rho_b = exc = vxc = vrho = wva = wvb = None

#    for i in range(nset):
#        vmat[0,i] = vmat[0,i] + vmat[0,i].conj().T
#        vmat[1,i] = vmat[1,i] + vmat[1,i].conj().T
    if isinstance(dma, np.ndarray) and dma.ndim == 2:
#        vmat = vmat[:,0]
        nelec = nelec.reshape(2)
        excsum = excsum[0]
    print(nelec, nelec_g)
    if trunc is not None:
        return nelec, excsum , excfsum
    else:
        return nelec, excsum 
                
def get_eta(rho1, rho2):
    ''' eta = (rho1/rho2)**(1/3)'''
    eta = (rho1 / rho2)**(1.0/3)
    for i in range(len(rho2)):
        if rho2[i] < 1e-45 or eta[i] < 1.0:
            #rho2[i] = 1e-45
            eta[i] = 1.0
    return eta

def get_f(rho, eta):
    b = [[-2.207193,     6.807648,     -6.386316,     2.860522,     -0.07466076],
         [ 1.128469,    -2.535669,      2.432821,    -1.064058,      0.03843687],
         [-0.2475593,    0.4243142,    -0.2565175,    0.08294749,   -0.003184296],
         [ 0.08616560,  -0.1715714,     0.1067547,   -0.02392882,    0.002579856],
         [-0.6500077e-2, 0.1714085e-1, -0.1462187e-1, 0.4423830e-2, -0.4427570e-3],
         [-0.2491486e-2, 0.5321373e-2, -0.3704699e-2, 0.9700054e-3, -0.9518308e-4]]
    x = (1.0/3) * np.log(3.0 / (4 * np.pi * rho))
    finv = np.zeros(rho.shape[0])
    for m in range(6):
        for n in range(5):
            finv += b[m][n] * x**m * eta**n
    f = finv**(-1)
    for i in range(len(rho)):
        #if rho[i] < 1e-45:
        #    #rho2[i] = 1e-45
        #    f[i] = 0.0
        #if eta[i] > 5.0:
        #    f[i] = 0.0
        if f[i] > 1.0 or x[i] > 5.0:
            f[i] = 1.0
    return f
