import numpy as np
import time
import psi4
from build_cabs import *
from stggtg import stggtg

class helper_mp2f12(object):

    def __init__(self, dimer, memory=8, algorithm='MO', reference='RHF'):
        
	self.conv = psi4.core.BasisSet.build(dimer,'BASIS', psi4.core.get_global_option('BASIS'))
	self.aux_basis = psi4.core.BasisSet.build(dimer,'DF_BASIS_MP2',"", "RIFIT", psi4.core.get_global_option('DF_BASIS_MP2'))
	self.scf_e, self.wfn = psi4.energy('SCF', return_wfn=True)
	
	self.cAO_CABS, self.cAUX_CABS = get_cabs(self.conv, self.aux_basis, self.wfn) 
	

	print("\nInitializing INTEGRALS object...\n")
        tinit_start = time.time()


        # Setup a few variables
        self.memory = memory
        self.nmo = self.wfn.nmo()
        self.ndocc = self.wfn.nalpha()
        self.nvirt = self.nmo - self.ndocc
	self.nfocc = self.wfn.nfrzc()
	self.naocc = self.ndocc - self.wfn.nfrzc()


	self.C = np.asarray(self.wfn.Ca())
	self.Co = np.asarray(self.wfn.Ca_subset("AO", "OCC"))
	self.Cactive = self.Co[:,self.nfocc:]
	self.Cv = np.asarray(self.wfn.Ca_subset("AO", "VIR"))
	self.eps = np.asarray(self.wfn.epsilon_a())


	# CABS
	self.ncabs = self.cAUX_CABS.shape[1]
	
        # Make slice, orbital, and size dictionaries
        self.slices = {
		       'i': slice(0, self.naocc), 
                       'o': slice(0, self.ndocc), 
                       'a': slice(self.ndocc, None),     
		       'p': slice(None, None),
                      }

        self.orbitals = {  'i': self.Cactive,
			   'o': self.Co,
                           'a': self.Cv,
			   'p': self.C,

                        }

        self.sizes = {  'i': self.naocc,
			'o': self.ndocc,
                        'a': self.nvirt,
			'p': self.nmo,
			'x': self.ncabs
		       }

   	self.basis = {	'i': self.conv,
			'o': self.conv,
	 	        'a': self.conv,
	 	        'p': self.conv,
		        'x': self.aux_basis,
		       }

        # Compute size of ERI tensor in GB
        self.dimer_wfn = psi4.core.Wavefunction.build(dimer, psi4.core.get_global_option('BASIS'))
        mints = psi4.core.MintsHelper(self.dimer_wfn.basisset())
        self.mints = mints
        ERI_Size = (self.nmo ** 4) * 8.e-9
        memory_footprint = ERI_Size * 4
        if memory_footprint > self.memory:
            psi4.core.clean()
            raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                            limit of %4.2f GB." % (memory_footprint, self.memory))

        # Integral generation from Psi4's MintsHelper
	self.Vao_oo = np.asarray(psi4.core.MintsHelper(self.dimer_wfn.basisset()).ao_potential())
        self.Vao_oa = np.asarray(psi4.core.MintsHelper(self.dimer_wfn.basisset()).ao_potential(self.conv,self.aux_basis))
        self.Vao_aa = np.asarray(psi4.core.MintsHelper(self.dimer_wfn.basisset()).ao_potential(self.aux_basis,self.aux_basis))

	self.Sao_oo = np.asarray(psi4.core.MintsHelper(self.dimer_wfn.basisset()).ao_overlap())
        self.Sao_oa = np.asarray(psi4.core.MintsHelper(self.dimer_wfn.basisset()).ao_overlap(self.conv,self.aux_basis))
        self.Sao_aa = np.asarray(psi4.core.MintsHelper(self.dimer_wfn.basisset()).ao_overlap(self.aux_basis,self.aux_basis))
	
	self.Tao_oo = np.asarray(psi4.core.MintsHelper(self.dimer_wfn.basisset()).ao_kinetic())
        self.Tao_oa = np.asarray(psi4.core.MintsHelper(self.dimer_wfn.basisset()).ao_kinetic(self.conv,self.aux_basis))
	self.Tao_aa = np.asarray(psi4.core.MintsHelper(self.dimer_wfn.basisset()).ao_kinetic(self.aux_basis,self.aux_basis))


    def get_size(self):
	return self.sizes


    def get_V(self):
        self.Vmo_oo = np.einsum('ui,vj,uv->ij', self.C, self.C, self.Vao_oo)
	self.Vmo_ox = np.einsum('ui,vj,uv->ij', self.C, self.cAO_CABS, self.Vao_oo)  + np.einsum('ui,vj,uv->ij', self.C, self.cAUX_CABS, self.Vao_oa)
	self.Vmo_xx = np.einsum('ui,vj,uv->ij', self.cAO_CABS, self.cAO_CABS, self.Vao_oo) + np.einsum('ui,vj,uv->ij', self.cAO_CABS, self.cAUX_CABS, self.Vao_oa) + np.einsum('ui,vj,vu->ij', self.cAUX_CABS, self.cAO_CABS, self.Vao_oa) + np.einsum('ui,vj,uv->ij', self.cAUX_CABS, self.cAUX_CABS, self.Vao_aa)
	return self.Vmo_oo, self.Vmo_ox , self.Vmo_xx

    
    def get_S(self):	
	self.Smo_oo = np.einsum('ui,vj,uv->ij', self.C, self.C, self.Sao_oo)
	self.Smo_ox = np.einsum('ui,vj,uv->ij', self.C, self.cAO_CABS, self.Sao_oo)  + np.einsum('ui,vj,uv->ij', self.C, self.cAUX_CABS, self.Sao_oa)
	self.Smo_xx = np.einsum('ui,vj,uv->ij', self.cAO_CABS, self.cAO_CABS, self.Sao_oo) + np.einsum('ui,vj,uv->ij', self.cAO_CABS, self.cAUX_CABS, self.Sao_oa) + np.einsum('ui,vj,vu->ij', self.cAUX_CABS, self.cAO_CABS, self.Sao_oa) + np.einsum('ui,vj,uv->ij', self.cAUX_CABS, self.cAUX_CABS, self.Sao_aa)

        return self.Smo_oo, self.Smo_ox , self.Smo_xx


    def get_T(self):
        self.Tmo_oo = np.einsum('ui,vj,uv->ij', self.C, self.C, self.Tao_oo)
        self.Tmo_ox = np.einsum('ui,vj,uv->ij', self.C, self.cAO_CABS, self.Tao_oo)  + np.einsum('ui,vj,uv->ij', self.C, self.cAUX_CABS, self.Tao_oa)
	self.Tmo_xx = np.einsum('ui,vj,uv->ij', self.cAO_CABS, self.cAO_CABS, self.Tao_oo) + np.einsum('ui,vj,uv->ij', self.cAO_CABS, self.cAUX_CABS, self.Tao_oa) + np.einsum('ui,vj,vu->ij', self.cAUX_CABS, self.cAO_CABS, self.Tao_oa) + np.einsum('ui,vj,uv->ij', self.cAUX_CABS, self.cAUX_CABS, self.Tao_aa)

	return self.Tmo_oo, self.Tmo_ox , self.Tmo_xx
		

        if self.alg == "AO":
            tstart = time.time()
            aux_basis = psi4.core.BasisSet.build(self.dimer_wfn.molecule(), "DF_BASIS_SCF",
                                            psi4.core.get_option("SCF", "DF_BASIS_SCF"),
                                            "JKFIT", psi4.core.get_global_option('BASIS'),
                                            puream=self.dimer_wfn.basisset().has_puream())

            self.jk = psi4.core.JK.build(self.dimer_wfn.basisset(), aux_basis)
            self.jk.set_memory(int(memory * 1e9))
            self.jk.initialize()
            print("\n...initialized JK objects in %5.2f seconds." % (time.time() - tstart))

        print("\n...finished initializing SAPT object in %5.2f seconds." % (time.time() - tinit_start))

    # Compute MO ERI tensor (g) on the fly
    def g(self, string):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('g: string %s does not have 4 elements' % string)

	count_aux_basis = len([i for i in string if i == 'x'])
	indexes_aux_basis = [i for i,j in enumerate(string) if j == 'x']

	Ioo = np.asarray(self.mints.ao_eri()).swapaxes(1,2)
	# ERI's from mints are of type (11|22) - need <12|12>

	if count_aux_basis == 0:
	    G = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], Ioo)
            G = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], G)
            G = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], G)
            G = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], G)
	
	elif count_aux_basis == 1:
            I = np.asarray(self.mints.ao_eri(self.basis[string[0]],self.basis[string[2]],self.basis[string[1]],self.basis[string[3]])).swapaxes(1,2)	
	    if string[0] == 'x':
	        G1 = np.einsum('pA,pqrs->Aqrs', self.cAO_CABS, Ioo)
	        G2 = np.einsum('pA,pqrs->Aqrs', self.cAUX_CABS, I)
	    else: 
		G1 = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], Ioo)
		G2 = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], I)
	    if string[1] == 'x':
	        G1 = np.einsum('qB,Aqrs->ABrs', self.cAO_CABS, G1)
		G2 = np.einsum('qB,Aqrs->ABrs', self.cAUX_CABS, G2)
	    else:
	        G1 = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], G1)
                G2 = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], G2)
	    if string[2] == 'x':
		G1 = np.einsum('rC,ABrs->ABCs', self.cAO_CABS, G1)
		G2 = np.einsum('rC,ABrs->ABCs', self.cAUX_CABS, G2)
	    else:
		G1 = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], G1)
                G2 = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], G2)
	
	    if string[3] == 'x':
	        G1 = np.einsum('sD,ABCs->ABCD', self.cAO_CABS, G1)
		G2 = np.einsum('sD,ABCs->ABCD', self.cAUX_CABS, G2)
	    else:
		G1 = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], G1)
                G2 = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], G2)
		

	
	    G = G1 + G2
	   

	elif count_aux_basis == 2:
	    Iaa = np.asarray(self.mints.ao_eri(self.basis[string[0]],self.basis[string[2]],self.basis[string[1]],self.basis[string[3]])).swapaxes(1,2)
	    ix1 = indexes_aux_basis[0]
	    ix2 = indexes_aux_basis[1]
	    self.newbasis = [] 
	    for i,j in enumerate(string):
	        if i == ix1: 
		    self.newbasis.append(self.aux_basis)
		else: self.newbasis.append(self.conv)
	    Iao = np.asarray(self.mints.ao_eri(self.newbasis[0],self.newbasis[2],self.newbasis[1],self.newbasis[3])).swapaxes(1,2)
		
	    self.newbasis2 = [] 
	    for i,j in enumerate(string):
                if i == ix2: 
		    self.newbasis2.append(self.aux_basis)
                else: self.newbasis2.append(self.conv)
            Ioa = np.asarray(self.mints.ao_eri(self.newbasis2[0],self.newbasis2[2],self.newbasis2[1],self.newbasis2[3])).swapaxes(1,2)


	    if string[0] == 'x':
                G1 = np.einsum('pA,pqrs->Aqrs', self.cAO_CABS, Ioo)
                G2 = np.einsum('pA,pqrs->Aqrs', self.cAO_CABS, Ioa)
		G3 = np.einsum('pA,pqrs->Aqrs', self.cAUX_CABS, Iao)
		G4 = np.einsum('pA,pqrs->Aqrs', self.cAUX_CABS, Iaa)
            else:
		G1 = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], Ioo)
                G2 = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], Ioa)
                G3 = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], Iao)
                G4 = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], Iaa)

	    if string[1] == 'x':
		G1 = np.einsum('qB,Aqrs->ABrs', self.cAO_CABS, G1)
                G4 = np.einsum('qB,Aqrs->ABrs', self.cAUX_CABS, G4)
		if string[0] == 'x':
			G2 = np.einsum('qB,Aqrs->ABrs', self.cAUX_CABS, G2)
			G3 = np.einsum('qB,Aqrs->ABrs', self.cAO_CABS, G3)
		else:
			G2 = np.einsum('qB,Aqrs->ABrs', self.cAO_CABS, G2)
                	G3 = np.einsum('qB,Aqrs->ABrs', self.cAUX_CABS, G3)

            else:
		G1 = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], G1)
                G2 = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], G2)
                G3 = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], G3)
                G4 = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], G4)

	    if string[2] == 'x':
		G1 = np.einsum('rC,ABrs->ABCs', self.cAO_CABS, G1)
                G4 = np.einsum('rC,ABrs->ABCs', self.cAUX_CABS, G4)
		if (string[0] == 'x') or (string[1] == 'x'):
			G2 = np.einsum('rC,ABrs->ABCs', self.cAUX_CABS, G2)
                	G3 = np.einsum('rC,ABrs->ABCs', self.cAO_CABS, G3)
		else:
			G2 = np.einsum('rC,ABrs->ABCs', self.cAO_CABS, G2)
                	G3 = np.einsum('rC,ABrs->ABCs', self.cAUX_CABS, G3)

            else:
		G1 = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], G1)
                G2 = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], G2)
                G3 = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], G3)
                G4 = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], G4)


            if string[3] == 'x':
		G1 = np.einsum('sD,ABCs->ABCD', self.cAO_CABS, G1)
		G2 = np.einsum('sD,ABCs->ABCD', self.cAUX_CABS, G2)
 	        G3 = np.einsum('sD,ABCs->ABCD', self.cAO_CABS, G3)
		G4 = np.einsum('sD,ABCs->ABCD', self.cAUX_CABS, G4)

	    else:
		G1 = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], G1)
                G2 = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], G2)
                G3 = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], G3)
                G4 = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], G4)

	    
            G = G1 + G2 + G3 + G4
	

	else:
            raise Exception('Illigal integral type.')


        return G


    def f12(self, string, gamma):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('f12: string %s does not have 4 elements' % string)

        count_aux_basis = len([i for i in string if i == 'x'])
        indexes_aux_basis = [i for i,j in enumerate(string) if j == 'x']

	coeffs, exps = stggtg(gamma)
        cf = psi4.core.CorrelationFactor(coeffs, exps)

        Ioo = np.asarray(self.mints.ao_f12(cf)).swapaxes(1,2)
        # F12 integrals from mints are of type (11|22) - need <12|12>

        if count_aux_basis == 0:
            G = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], Ioo)
            G = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], G)
            G = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], G)
            G = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], G)

        elif count_aux_basis == 1:
            I = np.asarray(self.mints.ao_f12(cf,self.basis[string[0]],self.basis[string[2]],self.basis[string[1]],self.basis[string[3]])).swapaxes(1,2)
            if string[0] == 'x':
                G1 = np.einsum('pA,pqrs->Aqrs', self.cAO_CABS, Ioo)
                G2 = np.einsum('pA,pqrs->Aqrs', self.cAUX_CABS, I)
            else:
                G1 = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], Ioo)
                G2 = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], I)
            if string[1] == 'x':
                G1 = np.einsum('qB,Aqrs->ABrs', self.cAO_CABS, G1)
                G2 = np.einsum('qB,Aqrs->ABrs', self.cAUX_CABS, G2)
            else:
                G1 = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], G1)
                G2 = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], G2)
            if string[2] == 'x':
                G1 = np.einsum('rC,ABrs->ABCs', self.cAO_CABS, G1)
                G2 = np.einsum('rC,ABrs->ABCs', self.cAUX_CABS, G2)
            else:
                G1 = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], G1)
                G2 = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], G2)

            if string[3] == 'x':
                G1 = np.einsum('sD,ABCs->ABCD', self.cAO_CABS, G1)
                G2 = np.einsum('sD,ABCs->ABCD', self.cAUX_CABS, G2)
            else:
                G1 = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], G1)
                G2 = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], G2)

            G = G1 + G2

	elif count_aux_basis == 2:
            Iaa = np.asarray(self.mints.ao_f12(cf,self.basis[string[0]],self.basis[string[2]],self.basis[string[1]],self.basis[string[3]])).swapaxes(1,2)
            ix1 = indexes_aux_basis[0]
            ix2 = indexes_aux_basis[1]
            self.newbasis = []
            for i,j in enumerate(string):
                if i == ix1:
                    self.newbasis.append(self.aux_basis)
                else: self.newbasis.append(self.conv)
            Iao = np.asarray(self.mints.ao_f12(cf,self.newbasis[0],self.newbasis[2],self.newbasis[1],self.newbasis[3])).swapaxes(1,2)

            self.newbasis = []
            for i,j in enumerate(string):
                if i == ix2:
                    self.newbasis.append(self.aux_basis)
                else: self.newbasis.append(self.conv)
            Ioa = np.asarray(self.mints.ao_f12(cf,self.newbasis[0],self.newbasis[2],self.newbasis[1],self.newbasis[3])).swapaxes(1,2)


            if string[0] == 'x':
                G1 = np.einsum('pA,pqrs->Aqrs', self.cAO_CABS, Ioo)
                G2 = np.einsum('pA,pqrs->Aqrs', self.cAO_CABS, Ioa)
                G3 = np.einsum('pA,pqrs->Aqrs', self.cAUX_CABS, Iao)
                G4 = np.einsum('pA,pqrs->Aqrs', self.cAUX_CABS, Iaa)
            else:
                G1 = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], Ioo)
                G2 = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], Ioa)
                G3 = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], Iao)
                G4 = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], Iaa)

            if string[1] == 'x':
                G1 = np.einsum('qB,Aqrs->ABrs', self.cAO_CABS, G1)
                G4 = np.einsum('qB,Aqrs->ABrs', self.cAUX_CABS, G4)
                if string[0] == 'x':
                        G2 = np.einsum('qB,Aqrs->ABrs', self.cAUX_CABS, G2)
                        G3 = np.einsum('qB,Aqrs->ABrs', self.cAO_CABS, G3)
                else:
                        G2 = np.einsum('qB,Aqrs->ABrs', self.cAO_CABS, G2)
                        G3 = np.einsum('qB,Aqrs->ABrs', self.cAUX_CABS, G3)

            else:
                G1 = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], G1)
                G2 = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], G2)
                G3 = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], G3)
                G4 = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], G4)

            if string[2] == 'x':
                G1 = np.einsum('rC,ABrs->ABCs', self.cAO_CABS, G1)
                G4 = np.einsum('rC,ABrs->ABCs', self.cAUX_CABS, G4)
                if (string[0] == 'x') or (string[1] == 'x'):
                        G2 = np.einsum('rC,ABrs->ABCs', self.cAUX_CABS, G2)
                        G3 = np.einsum('rC,ABrs->ABCs', self.cAO_CABS, G3)
                else:
                        G2 = np.einsum('rC,ABrs->ABCs', self.cAO_CABS, G2)
                        G3 = np.einsum('rC,ABrs->ABCs', self.cAUX_CABS, G3)

            else:
                G1 = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], G1)
                G2 = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], G2)
                G3 = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], G3)
                G4 = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], G4)


            if string[3] == 'x':
                G1 = np.einsum('sD,ABCs->ABCD', self.cAO_CABS, G1)
                G2 = np.einsum('sD,ABCs->ABCD', self.cAUX_CABS, G2)
                G3 = np.einsum('sD,ABCs->ABCD', self.cAO_CABS, G3)
                G4 = np.einsum('sD,ABCs->ABCD', self.cAUX_CABS, G4)

            else:
                G1 = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], G1)
                G2 = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], G2)
                G3 = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], G3)
                G4 = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], G4)

            G = G1 + G2 + G3 + G4

        else:
            raise Exception('Illigal integral type.')


        return G
	

    def f12squared(self, string, gamma):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('f12squared: string %s does not have 4 elements' % string)

        count_aux_basis = len([i for i in string if i == 'x'])
        indexes_aux_basis = [i for i,j in enumerate(string) if j == 'x']

        coeffs, exps = stggtg(gamma)
        cf = psi4.core.CorrelationFactor(coeffs, exps)

        Ioo = np.asarray(self.mints.ao_f12_squared(cf)).swapaxes(1,2)
        # F12squared Integrals from mints are of type (11|22) - need <12|12>

        if count_aux_basis == 0:
            G = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], Ioo)
            G = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], G)
            G = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], G)
            G = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], G)

        elif count_aux_basis == 1:
            I = np.asarray(self.mints.ao_f12_squared(cf,self.basis[string[0]],self.basis[string[2]],self.basis[string[1]],self.basis[string[3]])).swapaxes(1,2)
	    if string[0] == 'x':
                G1 = np.einsum('pA,pqrs->Aqrs', self.cAO_CABS, Ioo)
                G2 = np.einsum('pA,pqrs->Aqrs', self.cAUX_CABS, I)
            else:
                G1 = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], Ioo)
                G2 = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], I)
            if string[1] == 'x':
                G1 = np.einsum('qB,Aqrs->ABrs', self.cAO_CABS, G1)
                G2 = np.einsum('qB,Aqrs->ABrs', self.cAUX_CABS, G2)
            else:
                G1 = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], G1)
                G2 = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], G2)
            if string[2] == 'x':
                G1 = np.einsum('rC,ABrs->ABCs', self.cAO_CABS, G1)
                G2 = np.einsum('rC,ABrs->ABCs', self.cAUX_CABS, G2)
            else:
                G1 = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], G1)
                G2 = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], G2)

            if string[3] == 'x':
                G1 = np.einsum('sD,ABCs->ABCD', self.cAO_CABS, G1)
                G2 = np.einsum('sD,ABCs->ABCD', self.cAUX_CABS, G2)
            else:
                G1 = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], G1)
                G2 = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], G2)

            G = G1 + G2

        elif count_aux_basis == 2:
            Iaa = np.asarray(self.mints.ao_f12_squared(cf,self.basis[string[0]],self.basis[string[2]],self.basis[string[1]],self.basis[string[3]])).swapaxes(1,2)
            ix1 = indexes_aux_basis[0]
            ix2 = indexes_aux_basis[1]
            self.newbasis = []
            for i,j in enumerate(string):
                if i == ix1:
                    self.newbasis.append(self.aux_basis)
                else: self.newbasis.append(self.conv)
            Iao = np.asarray(self.mints.ao_f12_squared(cf,self.newbasis[0],self.newbasis[2],self.newbasis[1],self.newbasis[3])).swapaxes(1,2)

            self.newbasis = []
            for i,j in enumerate(string):
                if i == ix2:
                    self.newbasis.append(self.aux_basis)
                else: self.newbasis.append(self.conv)
            Ioa = np.asarray(self.mints.ao_f12_squared(cf,self.newbasis[0],self.newbasis[2],self.newbasis[1],self.newbasis[3])).swapaxes(1,2)


            if string[0] == 'x':
                G1 = np.einsum('pA,pqrs->Aqrs', self.cAO_CABS, Ioo)
                G2 = np.einsum('pA,pqrs->Aqrs', self.cAO_CABS, Ioa)
                G3 = np.einsum('pA,pqrs->Aqrs', self.cAUX_CABS, Iao)
                G4 = np.einsum('pA,pqrs->Aqrs', self.cAUX_CABS, Iaa)
            else:
                G1 = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], Ioo)
                G2 = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], Ioa)
                G3 = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], Iao)
                G4 = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], Iaa)

            if string[1] == 'x':
                G1 = np.einsum('qB,Aqrs->ABrs', self.cAO_CABS, G1)
                G4 = np.einsum('qB,Aqrs->ABrs', self.cAUX_CABS, G4)
                if string[0] == 'x':
                        G2 = np.einsum('qB,Aqrs->ABrs', self.cAUX_CABS, G2)
                        G3 = np.einsum('qB,Aqrs->ABrs', self.cAO_CABS, G3)
                else:
                        G2 = np.einsum('qB,Aqrs->ABrs', self.cAO_CABS, G2)
                        G3 = np.einsum('qB,Aqrs->ABrs', self.cAUX_CABS, G3)

            else:
                G1 = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], G1)
                G2 = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], G2)
                G3 = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], G3)
                G4 = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], G4)


            if string[2] == 'x':
                G1 = np.einsum('rC,ABrs->ABCs', self.cAO_CABS, G1)
                G4 = np.einsum('rC,ABrs->ABCs', self.cAUX_CABS, G4)
                if (string[0] == 'x') or (string[1] == 'x'):
                        G2 = np.einsum('rC,ABrs->ABCs', self.cAUX_CABS, G2)
                        G3 = np.einsum('rC,ABrs->ABCs', self.cAO_CABS, G3)
                else:
                        G2 = np.einsum('rC,ABrs->ABCs', self.cAO_CABS, G2)
                        G3 = np.einsum('rC,ABrs->ABCs', self.cAUX_CABS, G3)

            else:
                G1 = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], G1)
                G2 = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], G2)
                G3 = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], G3)
                G4 = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], G4)


            if string[3] == 'x':
                G1 = np.einsum('sD,ABCs->ABCD', self.cAO_CABS, G1)
                G2 = np.einsum('sD,ABCs->ABCD', self.cAUX_CABS, G2)
                G3 = np.einsum('sD,ABCs->ABCD', self.cAO_CABS, G3)
                G4 = np.einsum('sD,ABCs->ABCD', self.cAUX_CABS, G4)

            else:
                G1 = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], G1)
                G2 = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], G2)
                G3 = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], G3)
                G4 = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], G4)

            G = G1 + G2 + G3 + G4


        else:
            raise Exception('Illigal integral type.')

        return G

 

    def f12g12(self, string, gamma):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('f12g12: string %s does not have 4 elements' % string)

        coeffs, exps = stggtg(gamma)
        cf = psi4.core.CorrelationFactor(coeffs, exps)
        Ioo = np.asarray(self.mints.ao_f12g12(cf)).swapaxes(1,2)

        G = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], Ioo)
        G = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], G)
        G = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], G)
        G = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], G)

        return G



    def f12dc(self, string, gamma):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('f12dc: string %s does not have 4 elements' % string)

        coeffs, exps = stggtg(gamma)
        cf = psi4.core.CorrelationFactor(coeffs, exps)
        Ioo = np.asarray(self.mints.ao_f12_double_commutator(cf)).swapaxes(1,2)

        G = np.einsum('pA,pqrs->Aqrs', self.orbitals[string[0]], Ioo)
	G = np.einsum('qB,Aqrs->ABrs', self.orbitals[string[1]], G)
        G = np.einsum('rC,ABrs->ABCs', self.orbitals[string[2]], G)
        G = np.einsum('sD,ABCs->ABCD', self.orbitals[string[3]], G)

        return G



    # Grab epsilons
    def get_eps(self):
	return self.eps

# End MP2-F12 helper

class mp2f12_timer(object):
    def __init__(self, name):
        self.name = name
        self.start = time.time()
        print('\nStarting %s...' % name)

    def stop(self):
        t = time.time() - self.start
        print('...%s took a total of % .2f seconds.' % (self.name, t))


def mp2f12_printer(line, value):
    spacer = ' ' * (20 - len(line))
    print(line + spacer + '% 16.8f mH  % 16.8f kcal/mol' % (value * 1000, value * 627.509))
# End MP2-F12 helper

