from pyscf import scf
from pyscf import lib
from pyscf.lib import temporary_env, logger
from pyphf import util2
from functools import partial
import numpy as np
import scipy
from pyscf import df
from pyscf.ao2mo import _ao2mo
import ctypes

print = partial(print, flush=True)
einsum = partial(np.einsum, optimize=True)

def get_JKg(mol, Pg, no, X, hyb=None):
    Jg = []
    Kg = []
    Pg_ortho = []
    for pg in Pg:
        pg_ortho = einsum('ij,jk,lk->il', no, pg, no)
        #print(pg_ortho)
        Pg_ortho.append(pg_ortho)
    norb = int(Pg_ortho[0].shape[0]/2)
    Pgaa_ao = []
    Pgab_ao = []
    Pgba_ao = []
    Pgbb_ao = []
    for pg in Pg_ortho:
        pgaa = pg[:norb, :norb] # ortho ao
        #print(pgaa)
        pgab = pg[:norb, norb:]
        pgba = pg[norb:, :norb]
        pgbb = pg[norb:, norb:]
        # X . P(g) . X^H
        pgaa_ao = einsum('ij,jk,lk->il', X, pgaa, X) # regular ao
        #print(pgaa_ao)
        pgab_ao = einsum('ij,jk,lk->il', X, pgab, X)
        pgba_ao = einsum('ij,jk,lk->il', X, pgba, X)
        pgbb_ao = einsum('ij,jk,lk->il', X, pgbb, X)
        Pgaa_ao.append(pgaa_ao)
        Pgbb_ao.append(pgbb_ao)
        Pgab_ao.append(pgab_ao)
        Pgba_ao.append(pgba_ao)
    Pgaabb_ao = Pgaa_ao + Pgbb_ao
    #print(Pgaabb_ao.shape)
    ndm = len(Pgab_ao)
    vj,vk = scf.hf.get_jk(mol, Pgaabb_ao, hermi=0)
    #print(vj.shape)
    Jgaa_ao = vj[:ndm] + vj[ndm:] 
    Kgaa_ao = - vk[:ndm]
    Jgbb_ao = Jgaa_ao
    Kgbb_ao = - vk[ndm:]
    Kgab_ao = scf.hf.get_jk(mol, Pgab_ao, hermi=0)[1] *(-1)
    Kgba_ao = scf.hf.get_jk(mol, Pgba_ao, hermi=0)[1] *(-1)
    for i in range(len(Jgaa_ao)):
        jgaa = einsum('ji,jk,kl->il', X, Jgaa_ao[i], X)  # ortho ao
        kgaa = einsum('ji,jk,kl->il', X, Kgaa_ao[i], X)  # ortho ao
        #print(ggaa)
        kgab = einsum('ji,jk,kl->il', X, Kgab_ao[i], X) 
        kgba = einsum('ji,jk,kl->il', X, Kgba_ao[i], X) 
        jgbb = einsum('ji,jk,kl->il', X, Jgbb_ao[i], X) 
        kgbb = einsum('ji,jk,kl->il', X, Kgbb_ao[i], X) 
        jg = util2.stack22(jgaa, np.zeros(kgab.shape), 
                           np.zeros(kgba.shape), jgbb)
        kg = util2.stack22(kgaa, kgab, kgba, kgbb)
        jg_no = einsum('ji,jk,kl->il', no, jg, no)
        kg_no = einsum('ji,jk,kl->il', no, kg, no)
        if hyb is not None:
            kg_no = kg_no * hyb
        Jg.append(jg_no)
        Kg.append(kg_no)
    return Jg, Kg, Pg_ortho

def get_jk(mol, dm, hermi=1, opt=None):
    if opt is not None:
        opt = opt.get(None)
    return scf.hf.get_jk(mol, dm, hermi, opt)

def get_k(mol, dm, hermi=1, opt=None, omega=None):
    if opt is not None:
        opt = opt.get(omega)
        with temporary_env(opt, prescreen='CVHFnrs8_vk_prescreen'):
            vk = scf.hf.get_jk(mol, dm, hermi, opt, with_j=False, omega=omega)[1]
    else:
        vk = scf.hf.get_jk(mol, dm, hermi, opt, with_j=False, omega=omega)[1]

    return vk

def no2ortho(Pg, no):
    Pg_ortho = []
    for pg in Pg:
        pg_ortho = einsum('ij,jk,lk->il', no, pg, no)
        #print(pg_ortho)
        Pg_ortho.append(pg_ortho)
    return Pg_ortho

def get_Gg(mol, Pg, no, X, dm_last=None, Ggao_last=None, opt=None, hyb=None, rsh=None):
    Pg_ortho = no2ortho(Pg, no)
    Gg_ortho, Pgao, Ggao = get_Gg_ortho(mol, Pg_ortho, X, dm_last=dm_last, Ggao_last=Ggao_last, opt=opt, 
                                        hyb=hyb, rsh=rsh)
    Gg = ortho2no(Gg_ortho, no)
    return Gg, Gg_ortho, Pg_ortho, Pgao, Ggao


def get_Gg_ortho(mol, Pg_ortho, X, dm_last=None, Ggao_last=None, opt=None, with_df=None, 
                 hyb=None, rsh=None):
    #Pg_ortho = no2ortho(Pg)
    norb = int(Pg_ortho[0].shape[0]/2)
    Pgaa_ao = []
    Pgab_ao = []
    Pgba_ao = []
    Pgbb_ao = []
    for pg in Pg_ortho:
        pgaa = pg[:norb, :norb] # ortho ao
        #print(pgaa)
        pgab = pg[:norb, norb:]
        pgba = pg[norb:, :norb]
        pgbb = pg[norb:, norb:]
        # X . P(g) . X^H
        pgaa_ao = einsum('ij,jk,lk->il', X, pgaa, X) # regular ao
        #print(pgaa_ao)
        pgab_ao = einsum('ij,jk,lk->il', X, pgab, X)
        pgba_ao = einsum('ij,jk,lk->il', X, pgba, X)
        pgbb_ao = einsum('ij,jk,lk->il', X, pgbb, X)
        Pgaa_ao.append(pgaa_ao)
        Pgbb_ao.append(pgbb_ao)
        Pgab_ao.append(pgab_ao)
        Pgba_ao.append(pgba_ao)
    Pgaabb_ao = Pgaa_ao + Pgbb_ao
    Pgao = [Pgaabb_ao, Pgab_ao, Pgba_ao]
    if dm_last is not None:
        old_Pgaabb_ao, old_Pgab_ao, old_Pgba_ao = dm_last
        old_vj, old_vk, old_Ggab_ao, old_Ggba_ao = Ggao_last
    #print(Pgaabb_ao.shape)
    #nao = Pgaabb_ao.shape[-1]
    if rsh is not None:
        omega, alpha, hyb = rsh
    if with_df is not None:
        print('jk: rijk')
        vj,vk = get_jk_df_hermi0(with_df, Pgaabb_ao, hermi=0)
        Ggab_ao = get_jk_df_hermi0(with_df, Pgab_ao, hermi=0, with_j=False)[1]
        Ggba_ao = get_jk_df_hermi0(with_df, Pgba_ao, hermi=0, with_j=False)[1]
        if rsh is not None:
            vklr = get_jk_df_hermi0(with_df, Pgaabb_ao, hermi=0, with_j=False, omega=omega)[1]
            Ggab_ao_lr = get_jk_df_hermi0(with_df, Pgab_ao, hermi=0, with_j=False, omega=omega)[1]
            Ggba_ao_lr = get_jk_df_hermi0(with_df, Pgba_ao, hermi=0, with_j=False, omega=omega)[1]
    elif dm_last is None:
        print('jk: direct')
        #print('hermi', util2.is_hermi(Pgaabb_ao))
        vj,vk = get_jk(mol, Pgaabb_ao, hermi=0, opt=opt)
        Ggab_ao = get_k(mol, Pgab_ao, hermi=0, opt=opt)
        Ggba_ao = get_k(mol, Pgba_ao, hermi=0, opt=opt)
        if rsh is not None:
            vklr = get_k(mol, Pgaabb_ao, hermi=0, opt=opt, omega=omega)
            Ggab_ao_lr = get_k(mol, Pgab_ao, hermi=0, opt=opt, omega=omega)
            Ggba_ao_lr = get_k(mol, Pgba_ao, hermi=0, opt=opt, omega=omega)
        # print(vj[3][:4,:4])
        # print(vk[3][:4,:4])
        # print(Ggab_ao[3][:4,:4])
        # print(Ggba_ao[3][:4,:4])
    else:
        print('jk: direct(incfock)')
        d_aabb = util2.dmlist(Pgaabb_ao, old_Pgaabb_ao)
        d_ab = util2.dmlist(Pgab_ao, old_Pgab_ao)
        d_ba = util2.dmlist(Pgba_ao, old_Pgba_ao)
        #print('hermi', util2.is_hermi(d_aabb))
        vj,vk = get_jk(mol, d_aabb, hermi=0, opt=opt) 
        #vj = util2.dmlist(vj, old_vj, 1)
        #vk = util2.dmlist(vk, old_vk, 1)
        vj += old_vj
        vk += old_vk
        Ggab_ao = get_k(mol, d_ab, hermi=0, opt=opt) + old_Ggab_ao
        #Ggba_ao = get_k(mol, d_ba, hermi=0, opt=opt) + old_Ggba_ao
        tmp = get_k(mol, d_ba, hermi=0, opt=opt) 
        Ggba_ao = tmp + old_Ggba_ao
        #Ggab_ao = get_k(mol, Pgab_ao, hermi=0, opt=opt)
        #Ggba_ao = get_k(mol, Pgba_ao, hermi=0, opt=opt)
        # print(vj[3][:4,:4])
        # print(vk[3][:4,:4])
        # print(Ggab_ao[3][:4,:4])
        # print('ba, old, tmp, new')
        # print(old_Ggba_ao[3][:4,:4])
        # print(tmp[3][:4,:4])
        # print(Ggba_ao[3][:4,:4])
    ndm = len(Pgab_ao)
    #print(vj.shape)
    #print(vj)
    if hyb is not None:
        vk = vk * hyb
        Ggab_ao = Ggab_ao * hyb
        Ggba_ao = Ggba_ao * hyb
        if rsh is not None:
            vk = vk + vklr * (alpha-hyb)
            Ggab_ao = Ggab_ao + Ggab_ao_lr * (alpha-hyb)
            Ggba_ao = Ggba_ao + Ggba_ao_lr * (alpha-hyb)
    Ggaa_ao = vj[:ndm] + vj[ndm:] - vk[:ndm]
    Ggbb_ao = vj[:ndm] + vj[ndm:] - vk[ndm:]
    #Ggbb_ao = scf.hf.get_jk(mol, Pgbb_ao, hermi=0)
    #print(ggaa_ao)
    Ggao = [vj,vk,Ggab_ao, Ggba_ao]
    #ggbb_ao = scf.uhf.get_veff(mol, [pgaa_ao, pgbb_ao], hermi=0)[1]
        # X^H . G(g) . X
    Ggab_ao = Ggab_ao * (-1)
    Ggba_ao = Ggba_ao * (-1)
    Gg_ortho = []
    for i,ggab_ao in enumerate(Ggab_ao):
        #ggab_ao = Ggab_ao[i]
        ggba_ao = Ggba_ao[i]
        ggaa_ao = Ggaa_ao[i]
        ggbb_ao = Ggbb_ao[i]
        ggaa = einsum('ji,jk,kl->il', X, ggaa_ao, X)  # ortho ao
        #print(ggaa)
        ggab = einsum('ji,jk,kl->il', X, ggab_ao, X) 
        ggba = einsum('ji,jk,kl->il', X, ggba_ao, X) 
        ggbb = einsum('ji,jk,kl->il', X, ggbb_ao, X) 
        gg = util2.stack22(ggaa, ggab, ggba, ggbb)
        Gg_ortho.append(gg)
    return Gg_ortho, Pgao, Ggao


def ortho2no(Gg_ortho, no):    
    Gg = []
    for gg in Gg_ortho:
        gg_no = einsum('ji,jk,kl->il', no, gg, no)
        Gg.append(gg_no)
    return Gg

#class DF(df.DF):
DF = df.DF

def get_Gg_df(mol, Pg, no, X, dm_last=None, Ggao_last=None, opt=None, with_df=None, rsh=None):
    Pg_ortho = no2ortho(Pg, no)
    Gg_ortho, Pgao, Ggao = get_Gg_ortho(mol, Pg_ortho, X, #dm_last, Ggao_last, opt, 
                                        with_df=with_df, rsh=rsh)
    Gg = ortho2no(Gg_ortho, no)
    return Gg, Gg_ortho, Pg_ortho, Pgao, Ggao

get_jk_df = df.df_jk.get_jk
#get_k_df = df.df_jk.get_k

#def get_jk_df(dfobj, dm, hermi=0, with_j=True, with_k=True, direct_scf_tol=1e-13):
    
def get_jk_df_hermi0(dfobj, dm, hermi=0, with_j=True, with_k=True, direct_scf_tol=1e-13, omega=None):
    if omega is None:
        return _get_jk_df_hermi0(dfobj, dm, hermi, with_j, with_k, direct_scf_tol)
    with dfobj.range_coulomb(omega) as rsh_df:
        return _get_jk_df_hermi0(rsh_df, dm, hermi, with_j, with_k, direct_scf_tol)

def _get_jk_df_hermi0(dfobj, dm, hermi=0, with_j=True, with_k=True, direct_scf_tol=1e-13):
    assert (with_j or with_k)
    if (not with_k and not dfobj.mol.incore_anyway and
        # 3-center integral tensor is not initialized
        dfobj._cderi is None):
        return df.df_jk.get_j(dfobj, dm, hermi, direct_scf_tol), None

    t0 = t1 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(dfobj.stdout, dfobj.verbose)
    fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
    null = lib.c_null_ptr()

    dms = np.asarray(dm)
    dm_shape = dms.shape
    nao = dm_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]
    vj = 0
    vk = np.zeros_like(dms)

    if np.iscomplexobj(dms):
        raise NotImplementedError('Complex DM is not supported')
    
    if with_j:
        idx = np.arange(nao)
        dmtril = lib.pack_tril(dms + dms.conj().transpose(0,2,1))
        dmtril[:,idx*(idx+1)//2+idx] *= .5

    if not with_k:
        for eri1 in dfobj.loop():
            # uses numpy.matmul
            vj += dmtril.dot(eri1.T).dot(eri1)
    else:
        orbol, orbor = _decompose_rdm1_svd (None, dfobj.mol, dms)

        max_memory = dfobj.max_memory - lib.current_memory()[0]
        blksize = max(4, int(min(dfobj.blockdim, max_memory*.3e6/8/nao**2)))
        bufl = np.empty((blksize*nao,nao))
        bufr = np.empty((blksize*nao,nao))
        for eri1 in dfobj.loop(blksize):
            naux, nao_pair = eri1.shape
            assert (nao_pair == nao*(nao+1)//2)
            if with_j:
                # uses numpy.matmul
                vj += dmtril.dot(eri1.T).dot(eri1)

            for k in range(nset):
                nocc = orbol[k].shape[1]
                #print('nocc', nocc)
                if nocc > 0:
                    buf1l = bufl[:naux*nocc]
                    fdrv(ftrans, fmmm,
                         buf1l.ctypes.data_as(ctypes.c_void_p),
                         eri1.ctypes.data_as(ctypes.c_void_p),
                         orbol[k].ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(naux), ctypes.c_int(nao),
                         (ctypes.c_int*4)(0, nocc, 0, nao),
                         null, ctypes.c_int(0))
                    buf1r = bufr[:naux*nocc]
                    fdrv(ftrans, fmmm,
                         buf1r.ctypes.data_as(ctypes.c_void_p),
                         eri1.ctypes.data_as(ctypes.c_void_p),
                         orbor[k].ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(naux), ctypes.c_int(nao),
                         (ctypes.c_int*4)(0, nocc, 0, nao),
                         null, ctypes.c_int(0))
                    vk[k] += lib.dot(buf1l.T, buf1r)
            t1 = log.timer_debug1('jk', *t1)
    if with_j: vj = lib.unpack_tril(vj, 1).reshape(dm_shape)
    if with_k: vk = vk.reshape(dm_shape)
    logger.timer(dfobj, 'df vj and vk', *t0)
    return vj, vk

def _decompose_rdm1_svd (mf_grad, mol, dm):
    '''Decompose dms as U.Vh using SVD

    Args:
        mf_grad : instance of :class:`Gradients`
        mol : instance of :class:`gto.Mole`
        dm : ndarray or sequence of ndarrays of shape (nao,nao)
            Density matrices

    Returns:
        orbol : list of ndarrays of shape (nao,*)
            Contains non-null eigenvectors of density matrix
        orbor : list of ndarrays of shape (nao,*)
            Contains orbol * eigenvalues (occupancies)
    '''
    nao = mol.nao
    dms = np.asarray(dm).reshape (-1,nao,nao)
    orbor = []
    orbol = []
    for dm in dms:
        u, s, vh = _svd (dm)
        idx = np.abs (s)>1e-8
        orbol.append (np.asfortranarray (u[:,idx]))
        orbor.append (np.asfortranarray (einsum('i,ip->pi', s[idx], vh[idx])))

    return orbol, orbor

def _svd(a):
    try:
        u, s, vh = np.linalg.svd(a)
    except np.linalg.LinAlgError:
        print('Warning: SVD failed, using gesvd')
        u, s, vh = scipy.linalg.svd(a, lapack_driver='gesvd')
    return u, s, vh