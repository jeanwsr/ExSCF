import numpy as np
from pyphf import util2, wigner
import sympy
from sympy.physics.quantum.cg import CG
import os
from functools import partial
import time
from pyphf.timing import timing

print = partial(print, flush=True)
einsum = partial(np.einsum, optimize=True)

def verify_rdm12_relation(rdm1, rdm2, nelec):
    return np.allclose(rdm1, (nelec - 1)**-1 * rdm2.diagonal(axis1=-1, axis2=-2).sum(axis=-1))

def make_1pdm(suhf, Dg, dm_no, C_no):
    print('******* suhf density *****')
    t0 = time.time()
    no = suhf.no
    na,nb = suhf.nelec
    occ = na+nb
    C_oo = C_no[:occ,:occ]
    Ngg = get_Ngg(Dg, dm_no, occ)
    t1 = time.time()
    Pgg, Pgg_ortho = get_Pgg(Dg, dm_no, Ngg, occ, no)
    print(Pgg[0])
    suhf.Pgg_ortho = Pgg_ortho
    t2 = time.time()
    xgg = get_xgg(Ngg, C_oo)
    suhf.xgg = xgg
    t3 = time.time()
    if suhf.debug:
        print('time for Ngg: %.3f' % (t1-t0))
        print('time for Pgg: %.3f' % (t2-t1))
        print('time for xgg: %.3f' % (t3-t2))
    #wgtf0 = suhf.d
    S, Sz = suhf.S, suhf.Sz
    wght, cgf = do_cg(S, Sz, suhf.d, suhf.nbeta, suhf.grids)
    weight_f0, weight_fp1, weight_fp2 = wght
    cgfloat0, cgfloat1, cgfloat2, cgf1p1, cgf2p1, cgf2p2 = cgf

    xggint = suhf.integr_beta(np.array(xgg), fac='ci')
    print('xggint', xggint)

    X = suhf.X
    def ortho2reg(dm):
        return util2.reg2ortho(dm, X, False)

    norb = int(Pgg_ortho[0].shape[0]/2)
    Onepdm_a = []
    Onepdm_b = []
    Twopdm_aa = []
    Twopdm_bb = []
    Twopdm_ab = []
    for i,pgg_ortho in enumerate(Pgg_ortho):
        x = xgg[i]
        pggaa = pgg_ortho[:norb, :norb]
        pggab = pgg_ortho[:norb, norb:]
        pggba = pgg_ortho[norb:, :norb]
        pggbb = pgg_ortho[norb:, norb:]

        pggaa_ao_, pggab_ao_, pggba_ao_, pggbb_ao_ = list(map(ortho2reg, [pggaa, pggab, pggba, pggbb]))
        #print('xgg', x)
        #print('pgg\n', pggaa_ao, '\n', pggbb_ao, '\n', pggab_ao, '\n', pggba_ao)
        #print(pggbb_ao.dtype)
        #pgg_ao = util2.stack22(pggaa_ao, pggab_ao, pggba_ao, pggbb_ao)
        pgg1_ao = (pggaa_ao_ + pggbb_ao_)  * x / np.sqrt(2.0)
        pgg2_ao = (pggaa_ao_ - pggbb_ao_)  * x / np.sqrt(2.0)
        pggba_ao = pggba_ao_  * (-x)
        pggab_ao = pggab_ao_  * x
        if suhf.debug2:
            #print('pgg_ortho\n', pgg_ortho)
            print('pgg\n', pgg1_ao)
            print(pgg2_ao)
            print(pggba_ao)
            print(pggab_ao)
        #onepdm_a = pggaa_ao * x
        #onepdm_b = pggbb_ao * x
        # Here we assume S = Sz
        j2s1c = pgg1_ao * cgfloat0
        j2s2c = pgg2_ao * cgfloat1 
        if abs(Sz-1) <= S:
            j2s2c += pggba_ao * (-weight_fp1[i]*cgf1p1/weight_f0[i]) 
            #onepdm_a += pggab_ao * (-weight_fm1*cgf1m1/weight_fp0) * cgfloat1 / np.sqrt(2.0)
            #onepdm_b += -pggba_ao * (-weight_fp1[i]*cgf1p1/weight_f0[i]) * cgfloat1 / np.sqrt(2.0)
        onepdm_a = (j2s1c * cgfloat0 + j2s2c * cgfloat1) / np.sqrt(2.0)
        onepdm_b = (j2s1c * cgfloat0 - j2s2c * cgfloat1) / np.sqrt(2.0)
        if suhf.debug2:
            print('j2s1c, j2s2c\n', j2s1c* weight_f0[i], '\n', j2s2c* weight_f0[i])
        Onepdm_a.append(onepdm_a)
        Onepdm_b.append(onepdm_b)
        #print(onepdm_a.dtype)
        if suhf.do2pdm:
            twopdm = make_2pdm((pggaa_ao_, pggbb_ao_, pggba_ao_, pggab_ao_), 
                               x, norb, 
                               (cgfloat0, cgfloat1, cgfloat2, cgf1p1, cgf2p1, cgf2p2, 0.0, 0.0, 0.0),
                               (weight_f0[i], weight_fp1[i], 0.0, weight_fp2[i], 0.0)
                              )
            Twopdm_aa.append(twopdm[0])
            Twopdm_bb.append(twopdm[1])
            Twopdm_ab.append(twopdm[2])
    int1pdm_a = suhf.integr_beta(np.array(Onepdm_a), fac='ci') / xggint
    #print(type(int1pdm_a), int1pdm_a.dtype)
    int1pdm_b = suhf.integr_beta(np.array(Onepdm_b), fac='ci') / xggint
    #int1pdm_a = (int1pdm_a + int1pdm_a.T)/2.0
    #int1pdm_b = (int1pdm_b + int1pdm_b.T)/2.0
    if suhf.do2pdm:
        i2pdm_aa = suhf.integr_beta(np.array(Twopdm_aa), fac='ci') / xggint
        i2pdm_bb = suhf.integr_beta(np.array(Twopdm_bb), fac='ci') / xggint
        i2pdm_ab = suhf.integr_beta(np.array(Twopdm_ab), fac='ci') / xggint
        suhf.suhf_2pdm = (i2pdm_aa, i2pdm_ab, i2pdm_bb)
        if suhf.debug2:
            nmo = i2pdm_aa.shape[0]
            for i in range(nmo):
                for j in range(i,nmo):
                    for k in range(nmo):
                        for l in range(k,nmo):
                            print("aa %d %d %d %d %.6f" % (i,j,k,l,i2pdm_aa[i,j,k,l]))
            for i in range(nmo):
                for j in range(i,nmo):
                    for k in range(nmo):
                        for l in range(k,nmo):
                            print("bb %d %d %d %d %.6f" % (i,j,k,l,i2pdm_bb[i,j,k,l]))
    if suhf.debug:
        print('SUHF DM alpha\n', int1pdm_a, '\nSUHF DM beta\n', int1pdm_b)
    return [int1pdm_a, int1pdm_b]

def do_cg(S, Sz, d, nbeta, grids):
    print('S, Sz', S, Sz)
    weight_f0 = d
    #weight_f0 = wigner.wigner(S, Sz, suhf.nbeta, suhf.grids)[2]
    print('weight f0', weight_f0)
    #print('CG00, CG01')
    cgcoeff0, cgfloat0 = get_CG(S, Sz, 0, 0, S, Sz)
    cgcoeff1, cgfloat1 = get_CG(S, Sz, 1, 0, S, Sz)
    cgcoeff2, cgfloat2 = get_CG(S, Sz, 2, 0, S, Sz)
    #print(type(cgfloat1))
    if abs(Sz-1) <= S:
        print('weight fp1')
        weight_fp1 = wigner.wigner2(S*2,Sz-1, Sz, nbeta, grids)[2]
        print('CG2p1, CG1p1')
        _, cgf2p1 = get_CG(S, Sz, 2, -1, S, Sz-1)
        _, cgf1p1 = get_CG(S, Sz, 1, -1, S, Sz-1)
    else:
        weight_fp1 = weight_f0 * 0.0
        cgf1p1 = cgf2p1 = 0.0
    
    if abs(Sz-2) <= S:
        print('weight fp2')
        weight_fp2 = wigner.wigner2(S*2,Sz-2, Sz, nbeta, grids)[2]
        _, cgf2p2 = get_CG(S, Sz, 2, -2, S, Sz-2)
    else:
        weight_fp2 = weight_f0 * 0.0
        cgf2p2 = 0.0
    wght = weight_f0, weight_fp1, weight_fp2
    cgf = cgfloat0, cgfloat1, cgfloat2, cgf1p1, cgf2p1, cgf2p2
    return wght, cgf

@timing
def make_rdm12_no(suhf, debug=None):
    if suhf.debug:
        debug = True
    if debug is None:
        debug = False
    natorb = suhf.natorb[2]
    #print(np.dot(natorb.T, natorb))
    noinv = np.linalg.inv(natorb)
    rdm2 = suhf.suhf_2pdm
    #rdm2tot = rdm2[0] + rdm2[2] + rdm2[1] + rdm2[1].transpose(2,3,0,1)
    #rdm2tot = rdm2[0].transpose(0,2,1,3) + rdm2[2].transpose(0,2,1,3) + rdm2[1].transpose(0,2,1,3) + rdm2[1].transpose(1,3,0,2)
    #rdm2tot_no = einsum('pi, qj, ijkl, rk, sl -> pqrs', noinv, noinv, rdm2tot, noinv, noinv).transpose(0,3,1,2)
    def ao2mo(dm2):
        return  einsum('pi, qj, ijkl, rk, sl -> pqrs', noinv, noinv, dm2, noinv, noinv).transpose(0,3,1,2)*2.0
    rdm2_no = ao2mo(rdm2[0]), ao2mo(rdm2[1]), ao2mo(rdm2[2])
    rdm1_no = einsum('pi, tik, rk -> tpr', noinv, np.array(suhf.suhf_dm), noinv)
    #print(rdm1_no[:10,:10])
    if debug:
        print(rdm2_no[0][0,0,:10,:10])
        print(rdm2_no[0][0,:10,:10,0])
        print(rdm2_no[0][0,0,0,0], rdm2_no[0][0,0,1,1], rdm2_no[0][0,1,1,0])
    #nelec = suhf.nelec[0] + suhf.nelec[1]
    #rdm1_no2 = (nelec - 1)**-1 * rdm2_no.diagonal(axis1=-1, axis2=-2).sum(axis=-1)
    #print(rdm1_no2[:10,:10])
    #rdm2_no2 = make_2pdm_natorb(suhf)[0].transpose(0,3,1,2)
    #print(rdm2_no2[0,0,:10,:10])
    #print(rdm2_no2[0,:10,:10,0])
    #print(rdm2_no2[0,0,0,0], rdm2_no2[0,0,1,1], rdm2_no2[0,1,1,0])
    return rdm1_no, np.array(rdm2_no)

@timing
def make_rdm12_no_native(suhf, thresh=1e-5):
    natorb = suhf.natorb[2]
    #print(np.dot(natorb.T, natorb))
    noinv = np.linalg.inv(natorb)
    rdm1_no = einsum('pi, tik, rk -> tpr', noinv, np.array(suhf.suhf_dm), noinv)
    natocc = suhf.natocc[2]
    occ, [core, act, ext] = util2.dump_occ(natocc, 2.0, 1 - thresh)
    print('PDFT active threshold: ', thresh)
    print('PDFT active orb number: %d' % act)
    act_idx = slice(core, core+act)
    rdm2_no = make_2pdm_natorb(suhf, act_idx)
    def trans(dm2):
        return dm2.transpose(0,3,1,2)*2.0
    rdm2_no_trans = trans(rdm2_no[0]), trans(rdm2_no[1]), trans(rdm2_no[2])
    return rdm1_no, np.array(rdm2_no_trans), core, act_idx



def make_2pdm_natorb(suhf, act_idx):
    t0 = time.time()
    #no = suhf.no
    na,nb = suhf.nelec
    occ = na+nb
    #C_oo = C_no[:occ,:occ]
    #Ngg = get_Ngg(Dg, dm_no, occ)
    #Pgg, Pgg_ortho = get_Pgg(Dg, dm_no, Ngg, occ, no)
    Pgg_ortho = suhf.Pgg_ortho
    #xgg = get_xgg(Ngg, C_oo)
    xgg = suhf.xgg
    #wgtf0 = suhf.d
    S, Sz = suhf.S, suhf.Sz
    wght, cgf = do_cg(S, Sz, suhf.d, suhf.nbeta, suhf.grids)
    weight_f0, weight_fp1, weight_fp2 = wght
    cgfloat0, cgfloat1, cgfloat2, cgf1p1, cgf2p1, cgf2p2 = cgf

    xggint = suhf.integr_beta(np.array(xgg), fac='ci')
    print('xggint', xggint)

    X = suhf.X
    def ortho2reg(dm):
        return util2.reg2ortho(dm, X, False)
    natorb = suhf.natorb[2]
    noinv = np.linalg.inv(natorb)[act_idx,:]
    def reg2natorb(dm):
        return util2.reg2ortho(dm, noinv, False)

    norb = int(Pgg_ortho[0].shape[0]/2)
    Twopdm_aa = []
    Twopdm_bb = []
    Twopdm_ab = []
    for i,pgg_ortho in enumerate(Pgg_ortho):
        x = xgg[i]
        pggaa = pgg_ortho[:norb, :norb]
        pggab = pgg_ortho[:norb, norb:]
        pggba = pgg_ortho[norb:, :norb]
        pggbb = pgg_ortho[norb:, norb:]

        pggaa_ao_, pggab_ao_, pggba_ao_, pggbb_ao_ = list(map(ortho2reg, [pggaa, pggab, pggba, pggbb]))
        pggaa_no_, pggab_no_, pggba_no_, pggbb_no_ = list(map(reg2natorb, [pggaa_ao_, pggab_ao_, pggba_ao_, pggbb_ao_]))
        twopdm = make_2pdm((pggaa_no_, pggbb_no_, pggba_no_, pggab_no_), 
                               x, norb, 
                               (cgfloat0, cgfloat1, cgfloat2, cgf1p1, cgf2p1, cgf2p2, 0.0, 0.0, 0.0),
                               (weight_f0[i], weight_fp1[i], 0.0, weight_fp2[i], 0.0)
                              )
        Twopdm_aa.append(twopdm[0])
        Twopdm_bb.append(twopdm[1])
        Twopdm_ab.append(twopdm[2])
    #print(pggaa_no_[0])
    i2pdm_aa = suhf.integr_beta(np.array(Twopdm_aa), fac='ci') / xggint
    i2pdm_bb = suhf.integr_beta(np.array(Twopdm_bb), fac='ci') / xggint
    i2pdm_ab = suhf.integr_beta(np.array(Twopdm_ab), fac='ci') / xggint
    suhf.suhf_2pdm_no = (i2pdm_aa, i2pdm_ab, i2pdm_bb)
    #natocc = suhf.natocc[2]
    #occ, [core, act, ext] = util2.dump_occ(natocc, 2.0, 0.99999)
    if suhf.debug:
    #    nmo = i2pdm_aa.shape[0]
    #    for i in range(core, core+act):
    #        for j in range(i,core+act):
    #            for k in range(core, core+act):
    #                for l in range(k,core+act):
    #                    print("aa %d %d %d %d %.6f" % (i,j,k,l,i2pdm_aa[i,j,k,l]))
        print(i2pdm_aa[0,0])
        print(suhf.suhf_dm[0])
    return suhf.suhf_2pdm_no

def contr1(p1,p2,p3,p4, fac):
    j2p0 = einsum('il,jk->ijkl', p1, p2)
    j2p0 -= einsum('ik,jl->ijkl', p3, p4)
    #j2p0 = einsum('li,kj->ijkl', p1, p2)
    #j2p0 -= einsum('ki,lj->ijkl', p3, p4)
    j2p0 *= fac
    return j2p0

def make_2pdm(pgg, x, norb, cgf, wgt):
    pggaa, pggbb, pggba, pggab = pgg
    cgf00, cgf10, cgf20, cgf1p1, cgf2p1, cgf2p2, cgf1m1, cgf2m1, cgf2m2 = cgf
    wgtf0, wgtfp1, wgtfm1, wgtfp2, wgtfm2 = wgt
    wgtfp1 /= wgtf0
    wgtfm1 /= wgtf0
    wgtfp2 /= wgtf0
    wgtfm2 /= wgtf0
    
    j2aaaa = contr1(pggaa, pggaa, pggaa, pggaa, x/2.0)
    j2bbbb = contr1(pggbb, pggbb, pggbb, pggbb, x/2.0)
    j2abba = contr1(pggaa, pggbb, pggba, pggab, x/2.0)
    j2baab = contr1(pggbb, pggaa, pggab, pggba, x/2.0)
    j2abab = contr1(pggba, pggab, pggaa, pggbb, x/2.0)
    j2baba = contr1(pggab, pggba, pggbb, pggaa, x/2.0)
    j2aabb = contr1(pggba, pggba, pggba, pggba, x/2.0)
    j2bbaa = contr1(pggab, pggab, pggab, pggab, x/2.0)
    j2aaab = contr1(pggba, pggaa, pggaa, pggba, x/2.0)
    j2aaba = contr1(pggaa, pggba, pggba, pggaa, x/2.0)
    j2abaa = contr1(pggaa, pggab, pggaa, pggab, x/2.0)
    j2baaa = contr1(pggab, pggaa, pggab, pggaa, x/2.0)
    j2bbba = contr1(pggab, pggbb, pggbb, pggab, x/2.0)
    j2bbab = contr1(pggbb, pggab, pggab, pggbb, x/2.0)
    j2babb = contr1(pggbb, pggba, pggbb, pggba, x/2.0)
    j2abbb = contr1(pggba, pggbb, pggba, pggbb, x/2.0)
    #print('aaaa',j2aaaa[0,2,0,2])
    #print('bbbb',j2bbbb[0,2,0,2])
    #print('abba',j2abba[0,2,0,2])
    #print('aaba',j2aaba[0,2,0,2])
    #print('babb',j2babb[0,2,0,2])
    j2t10 = (j2aaaa + j2bbbb + j2abba + j2baab)*0.5
    
    j2t20 = (j2aaaa - j2bbbb - j2abba + j2baab)*0.5
    j2t2p1 = (j2aaba + j2babb)*(-1/np.sqrt(2.0))
    j2t2m1 = (j2abaa + j2bbab)*(+1/np.sqrt(2.0))
    j2t30 = (j2aaaa - j2bbbb + j2abba - j2baab)*0.5
    j2t3p1 = (j2aaab + j2abbb)*(-1/np.sqrt(2.0))
    j2t3m1 = (j2baaa + j2bbba)*(+1/np.sqrt(2.0))
    j2t40 = (-j2aaaa - j2bbbb + j2abba + j2baab)/(2*np.sqrt(3.0)) + (j2abab + j2baba)*(-1/np.sqrt(3.0))
    j2t60 = (j2aaaa + j2bbbb - j2abba - j2baab - j2abab - j2baba) / np.sqrt(6.0)
    j2t6p1 = (-j2aaab + j2abbb - j2aaba + j2babb)*0.5
    j2t6m1 = (j2baaa - j2bbba + j2abaa - j2bbab)*0.5
    j2t6p2 = j2aabb
    j2t6m2 = j2bbaa
    j2t1c = j2t10 * cgf00
    j2t2c = j2t20 * cgf10 - j2t2p1 * cgf1p1 * wgtfp1 #- j2t2m1 * cgf1m1 * wgtfm1
    j2t3c = j2t30 * cgf10 - j2t3p1 * cgf1p1 * wgtfp1 #- j2t3m1 * cgf1m1 * wgtfm1
    j2t4c = j2t40 * cgf00
    j2t6c = j2t60 * cgf20 - j2t6p1 * cgf2p1 * wgtfp1 #- j2t6m1 * cgf2m1 * wgtfm1 
    j2t6c += j2t6p2 * cgf2p2 * wgtfp2 #+ j2t6m2 * cgf2m2 * wgtfm2
    #print('j2t20, j2t2p1', j2t20[0,2,0,2], j2t2p1[0,2,0,2])
    #print(j2t1c[0,2,0,2]*wgtf0)
    #print(j2t4c[0,2,0,2]*wgtf0)
    #print(j2t2c[0,2,0,2]*wgtf0)
    #print(j2t3c[0,2,0,2]*wgtf0)
    #print(j2t6c[0,2,0,2]*wgtf0)
    # 2pdm aaaa, bbbb, aabb
    j2pdm0 = j2t1c * cgf00/2.0 - j2t4c * cgf00 / (2.0*np.sqrt(3.0)) + j2t2c * cgf10/2.0 + j2t3c * cgf10/2.0 + j2t6c * cgf20/np.sqrt(6.0)
    j2pdm1 = j2t1c * cgf00/2.0 - j2t4c * cgf00 / (2.0*np.sqrt(3.0)) - j2t2c * cgf10/2.0 - j2t3c * cgf10/2.0 + j2t6c * cgf20/np.sqrt(6.0)
    j2pdm2 = j2t1c * cgf00/2.0 + j2t4c * cgf00 / (2.0*np.sqrt(3.0)) - j2t2c * cgf10/2.0 + j2t3c * cgf10/2.0 - j2t6c * cgf20/np.sqrt(6.0)
    #print('j2pdm0', j2pdm0[0,2,0,2]*wgtf0)
    return j2pdm0, j2pdm1, j2pdm2
    


def natorb(suhf, dm):
    #print(type(dm[0]), dm[0].dtype)
    XS = suhf.XS
    X = suhf.X
    dmab = dm[0] + dm[1]
    dm_s = dm[0] - dm[1]
    # P = S^0.5 * dm * S^0.5
    P0 = einsum('ij,jk,lk->il', XS, dm[0], XS)
    P1 = einsum('ij,jk,lk->il', XS, dm[1], XS)
    Pab = P0 + P1
    P_s = P0 - P1
    natocc_a, natorb_a = np.linalg.eigh(-P0)
    natocc_b, natorb_b = np.linalg.eigh(-P1)
    natocc_ab, natorb_ab = np.linalg.eigh(-Pab)
    natocc_s, natorb_s = np.linalg.eigh(-P_s)
    natorb_a = np.dot(X, natorb_a)
    natorb_b = np.dot(X, natorb_b)
    natorb_ab = np.dot(X, natorb_ab)
    natorb_s = np.dot(X, natorb_s)
    natocc_a = -1 * natocc_a
    natocc_b = -1 * natocc_b
    natocc_ab = -1 * natocc_ab
    natocc_s = -1 * natocc_s
    if suhf.debug:
        print('SUHF natural orbitals alpha\n', natorb_a)
        print('SUHF natural orbitals beta\n', natorb_b)
        print('SUHF natural orbitals, total\n', natorb_ab)
    dump_occ = util2.dump_occ
    print('SUHF NO occ alpha: ', dump_occ(natocc_a, 1.0)[0])
    print('SUHF NO occ beta:  ', dump_occ(natocc_b, 1.0)[0])
    occ_ab, [core, act, ext] = dump_occ(natocc_ab, 2.0)
    print('SUHF NO occ total: ', occ_ab)
    #print('SUHF NO occ spin: ', natocc_s)
    print('core %d, active %d, external %d' % (core, act, ext))
    suhf.core = core
    suhf.act = act
    return [natorb_a, natorb_b, natorb_ab], [natocc_a, natocc_b, natocc_ab]

def get_CG(j1, m1, j2, m2, j, m):
    j1, m1, j2, m2, j, m = list(map(sympy.Rational, [j1, m1, j2, m2, j, m]))
    CGfunc = CG(j1, m1, j2, m2, j, m)
    CGcoeff =  CGfunc.doit()
    CGfloat = float(CGcoeff.evalf())
    print('Clebsch-Gordan coeff: < {} {} {} {} | {} {} > = {} = {:.6f}'.format(j1,m1,j2,m2,j,m,CGcoeff, CGfloat))
    return CGcoeff, CGfloat


def get_Ngg(Dg, dm_no, occ):
    Ngg = []
    for dg in Dg:
        ngg_inv = einsum('ij,jl,lm->im', dm_no[:occ,:], dg, dm_no[:,:occ])
        ngg = np.linalg.inv(ngg_inv)
        Ngg.append(ngg)
        #print('Ngg')
        #print(ngg)
    return Ngg

def get_Pgg(Dg, dm_no, Ngg, occ, no):
    Pgg = []
    Pgg_ortho = []
    #print('no\n', no)
    for i, dg in enumerate(Dg):
        pgg = einsum('ij,jk,kl,ln->in', dg, dm_no[:,:occ], Ngg[i], dm_no[:occ,:])
        #pgg = einsum('ij,kj->ik', pgg, dg)
        Pgg.append(pgg)
        #print('pgg')
        #print(pgg)
        pgg_ortho = einsum('ij,jk,lk->il', no, pgg, no)
        Pgg_ortho.append(pgg_ortho)
    return Pgg, Pgg_ortho

def get_xgg(Ngg, C_oo):
    detC = np.linalg.det(C_oo)
    xgg = []
    for ngg in Ngg:
        detn = np.linalg.det(ngg)
        x = 1.0 / (detC * detn * detC)
        xgg.append(x)
    print('xgg', xgg)
    return xgg
