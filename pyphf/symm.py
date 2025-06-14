from pyphf import suscf
from pyscf import scf, symm, lib
from pyscf.scf import hf_symm
import numpy as np
import scipy

eigh = scipy.linalg.eigh

class SymAdaptedSUHF(suscf.SUHF):
    def __init__(self, guesshf=None):
        super().__init__(guesshf=guesshf)
        self.symm = True
        self.irrep_nelec = {}

    def build(self):
        super().build()
        if self.guesshf is not None:
            self.irrep_nelec = self.guesshf.get_irrep_nelec()

    def get_orbsym(self, mo_coeff):
        #s = np.eye(mo_coeff[0].shape[0])
        s = self.ovlp
        mo_reg = self.mo_ortho2reg(mo_coeff)
        return scf.uhf_symm.get_orbsym(self.mol, mo_reg, s)

    #orbsym = property(get_orbsym)
    
    def get_occ(self, mo_e=None, mo_coeff=None):
        ''' We assumed mo_energy are grouped by symmetry irreps, (see function
        self.eig). The orbitals are sorted after SCF.
        '''
        if mo_e is None: 
            #mo_e = self.mo_e
            return suscf.get_occ(self)
        mol = self.mol
        if not mol.symmetry:
            raise RuntimeError('mol.symmetry not enabled')

        orbsym = self.get_orbsym(mo_coeff)
        self.orbsym = orbsym
        orbsyma, orbsymb = orbsym
        #print('orbsym_a = ', orbsyma)
        #print('orbsym_b = ', orbsymb)
        mo_occ = np.zeros_like(mo_e)
        idx_ea_left = []
        idx_eb_left = []
        neleca_fix = nelecb_fix = 0
        print(f'irrep_nelec = {self.irrep_nelec}')
        for i, ir in enumerate(mol.irrep_id):
            irname = mol.irrep_name[i]
            ir_idxa = np.where(orbsyma == ir)[0]
            ir_idxb = np.where(orbsymb == ir)[0]
            if irname in self.irrep_nelec:
                if isinstance(self.irrep_nelec[irname], (int, np.integer)):
                    nelecb = self.irrep_nelec[irname] // 2
                    neleca = self.irrep_nelec[irname] - nelecb
                else:
                    neleca, nelecb = self.irrep_nelec[irname]
                ea_idx = np.argsort(mo_e[0][ir_idxa].round(9), kind='stable')
                eb_idx = np.argsort(mo_e[1][ir_idxb].round(9), kind='stable')
                mo_occ[0,ir_idxa[ea_idx[:neleca]]] = 1
                mo_occ[1,ir_idxb[eb_idx[:nelecb]]] = 1
                neleca_fix += neleca
                nelecb_fix += nelecb
            else:
                idx_ea_left.append(ir_idxa)
                idx_eb_left.append(ir_idxb)

        nelec = self.nelec
        neleca_float = nelec[0] - neleca_fix
        nelecb_float = nelec[1] - nelecb_fix
        assert (neleca_float >= 0)
        assert (nelecb_float >= 0)
        if len(idx_ea_left) > 0:
            idx_ea_left = np.hstack(idx_ea_left)
            ea_left = mo_e[0][idx_ea_left]
            ea_sort = np.argsort(ea_left.round(9), kind='stable')
            occ_idx = idx_ea_left[ea_sort][:neleca_float]
            mo_occ[0][occ_idx] = 1
        if len(idx_eb_left) > 0:
            idx_eb_left = np.hstack(idx_eb_left)
            eb_left = mo_e[1][idx_eb_left]
            eb_sort = np.argsort(eb_left.round(9), kind='stable')
            occ_idx = idx_eb_left[eb_sort][:nelecb_float]
            mo_occ[1][occ_idx] = 1

        vir_idx = (mo_occ[0]==0)
        if self.verbose >= 0 and np.count_nonzero(vir_idx) > 0:
            noccsa = []
            noccsb = []
            for i, ir in enumerate(mol.irrep_id):
                irname = mol.irrep_name[i]
                ir_idxa = orbsyma == ir
                ir_idxb = orbsymb == ir
                noccsa.append(np.count_nonzero(mo_occ[0][ir_idxa]))
                noccsb.append(np.count_nonzero(mo_occ[1][ir_idxb]))

            ir_id2name = dict(zip(mol.irrep_id, mol.irrep_name))
            ehomo = ehomoa = max(mo_e[0][mo_occ[0]>0 ])
            elumo = elumoa = min(mo_e[0][mo_occ[0]==0])
            irhomoa = ir_id2name[orbsyma[mo_e[0] == ehomoa][0]]
            irlumoa = ir_id2name[orbsyma[mo_e[0] == elumoa][0]]
            #logger.info(self, 'alpha HOMO (%s) = %.15g  LUMO (%s) = %.15g',
            #            irhomoa, ehomoa, irlumoa, elumoa)
            print(f'irrep: alpha HOMO {irhomoa} LUMO {irlumoa}')
            if nelecb_float > 0:
                ehomob = max(mo_e[1][mo_occ[1]>0 ])
                elumob = min(mo_e[1][mo_occ[1]==0])
                irhomob = ir_id2name[orbsymb[mo_e[1] == ehomob][0]]
                irlumob = ir_id2name[orbsymb[mo_e[1] == elumob][0]]
                #logger.info(self, 'beta  HOMO (%s) = %.15g  LUMO (%s) = %.15g',
                #            irhomob, ehomob, irlumob, elumob)
                print(f'irrep: beta HOMO {irhomob} LUMO {irlumob}')
                ehomo = max(ehomoa,ehomob)
                elumo = min(elumoa,elumob)

            print('alpha irrep_nelec = %s'% noccsa)
            print('beta  irrep_nelec = %s'% noccsb)
            #hf_symm._dump_mo_energy(mol, mo_e[0], mo_occ[0], ehomo, elumo,
            #                        orbsyma, 'alpha-', verbose=4)
            #hf_symm._dump_mo_energy(mol, mo_e[1], mo_occ[1], ehomo, elumo,
            #                        orbsymb, 'beta-', verbose=4)

        #     if mo_coeff is not None and self.verbose >= logger.DEBUG:
        #         ovlp_ao = self.get_ovlp()
        #         ss, s = self.spin_square((mo_coeff[0][:,mo_occ[0]>0],
        #                                   mo_coeff[1][:,mo_occ[1]>0]), ovlp_ao)
        #         logger.debug(self, 'multiplicity <S^2> = %.8g  2S+1 = %.8g', ss, s)
        return mo_occ

    def Diag_Feff(self, F):
        mol = self.mol
        nirrep = mol.symm_orb.__len__()
        #s = mol.intor('int1e_ovlp')
        s = self.ovlp
        s = symm.symmetrize_matrix(s, mol.symm_orb)
        F_reg = self.f_ortho2reg(F)
        #print('F, ao/so')
        #print(F_reg[1])
        ha = symm.symmetrize_matrix(F_reg[0], mol.symm_orb)
        hb = symm.symmetrize_matrix(F_reg[1], mol.symm_orb)
        #print(hb)
        cs_b = []
        cs_a = []
        es_b = []
        es_a = []
        orbsym_a = []
        orbsym_b = []
        if mol.groupname in ('Dooh', 'Coov'):
            for ir in range(nirrep):
                irrep_id = mol.irrep_id[ir]
                irrep_1d = irrep_id in (0, 1, 4, 5)
                irrep_2dx = irrep_id % 2 == 0
                if irrep_1d or irrep_2dx:
                    ea, ca = eigh(ha[ir], s[ir])
                    eb, cb = eigh(hb[ir], s[ir])
                    cs_a.append(ca)
                    cs_b.append(cb)
                    es_a.append(ea)
                    es_b.append(eb)
                    orbsym_a.append([mol.irrep_id[ir]] * ea.size)
                    orbsym_b.append([mol.irrep_id[ir]] * eb.size)

                if not irrep_1d and irrep_2dx:
                    # force 2D irreps using the same coefficients
                    irrep_conj = irrep_id ^ 1
                    assert mol.irrep_id[ir+1] == irrep_conj
                    cs_a.append(ca)
                    cs_b.append(cb)
                    es_a.append(ea)
                    es_b.append(eb)
                    orbsym_a.append([irrep_conj] * ea.size)
                    orbsym_b.append([irrep_conj] * eb.size)
        else:
            for ir in range(nirrep):
                ea, ca = eigh(ha[ir], s[ir])
                eb, cb = eigh(hb[ir], s[ir])
                cs_a.append(ca)
                cs_b.append(cb)
                es_a.append(ea)
                es_b.append(eb)
                orbsym_a.append([mol.irrep_id[ir]] * ea.size)
                orbsym_b.append([mol.irrep_id[ir]] * eb.size)

        #print(cs_a)
        ea = np.hstack(es_a)
        eb = np.hstack(es_b)
        ca = hf_symm.so2ao_mo_coeff(mol.symm_orb, cs_a)
        #print(ca)
        ca = self.mo_reg2ortho(ca)
        #print(ca)
        ca = lib.tag_array(ca, orbsym=np.hstack(orbsym_a))
        cb = hf_symm.so2ao_mo_coeff(mol.symm_orb, cs_b)
        cb = self.mo_reg2ortho(cb)
        cb = lib.tag_array(cb, orbsym=np.hstack(orbsym_b))
        #return np.asarray((ea,eb)), (ca,cb)
        return [ea,eb], [ca,cb]
