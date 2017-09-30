from stggtg import stggtg
from psi4 import MintsHelper, CorrelationFactor
import numpy as np

# Orbital labels
orbitals = {'f':'focc', 
	    'a':'aocc', 
            'o':'occ', 
            'v':'vir', 
            'r':'act', 
	    'p':'obs',
    	    'x':'cabs'}

# Available integrals
one_electron_integrals = ('s', 't', 'v')
two_electron_integrals = ('g', 'f12', 'f12g12', 'f12_squared', 'f12_double_commutator')


class Integrals:
	def __init__(self, bs, gamma = 0.0):
		#self.string = string
		self.bs_ = bs
		self.gamma = gamma
		self.mints_ = MintsHelper(bs['obs'].basisset())


	def get_basis(self, label):
        	"""
        	Returns a list of the basis sets corresponding to the labels.
        	"""
        	bs = []
        	for s in label:
            		if not s in orbitals:
                		raise Exception('Unknown orbital label!')
            		keyword = orbitals[s]
            		bs.append(self.bs_[keyword])

		return bs

	def TransformToMo(self, type_integral, I,  bs):
       		if type_integral in one_electron_integrals:
           		I = np.einsum('pq,pP->Pq', I, bs[0].C())
           		I = np.einsum('Pq,qQ->PQ', I, bs[1].C())
           	elif type_integral in two_electron_integrals:
            		I = np.einsum('pqrs,pP->Pqrs', I, bs[0].C())
            		I = np.einsum('Pqrs,qQ->PQrs', I, bs[1].C())
            		I = np.einsum('PQrs,rR->PQRs', I, bs[2].C())
            		I = np.einsum('PQRs,sS->PQRS', I, bs[3].C())
          	else:
           		raise Exception('Unknown integral type.')

		return I


	def s(self, string):
		baslist = self.get_basis(string)
	    	b = [i.basisset() for i in baslist]
		
		I = np.array(self.mints_.ao_overlap(b[0],b[1]))
            	Imo = self.TransformToMo('s', I, baslist)

        	return Imo


	def t(self, string):
		baslist = self.get_basis(string)
		b = [i.basisset() for i in baslist]

                I = np.array(self.mints_.ao_kinetic(b[0],b[1]))
                Imo = self.TransformToMo('t', I, baslist)

                return Imo


	def v(self, string):
                baslist = self.get_basis(string)
                b = [i.basisset() for i in baslist]

                I = np.array(self.mints_.ao_potential(b[0],b[1]))
                Imo = self.TransformToMo('v', I, baslist)

                return Imo


	def g(self, string):
                baslist = self.get_basis(string)
                b = [i.basisset() for i in baslist]
		#I = np.array(self.mints_.ao_eri(b[0],b[2],b[1],b[3]))
		I = np.array(self.mints_.ao_eri(b[0],b[2],b[1],b[3])).swapaxes(1, 2)
		Imo = self.TransformToMo('g', I, baslist)

                return Imo

	
	def f12(self, string):
		coeffs, exps = stggtg(self.gamma)
		cf = CorrelationFactor(coeffs, exps)
                baslist = self.get_basis(string)
                b = [i.basisset() for i in baslist]

		#I = np.array(self.mints_.ao_f12(cf,b[0],b[2],b[1],b[3]))
                I = np.array(self.mints_.ao_f12(cf,b[0],b[2],b[1],b[3])).swapaxes(1, 2)
                Imo = self.TransformToMo('f12', I, baslist)

                return Imo


	def f12g12(self, string):
                coeffs, exps = stggtg(self.gamma)
                cf = CorrelationFactor(coeffs, exps)
                baslist = self.get_basis(string)
                b = [i.basisset() for i in baslist]

                I = np.array(self.mints_.ao_f12g12(cf)).swapaxes(1, 2)
                Imo = self.TransformToMo('f12g12', I, baslist)

                return Imo	

	
	def f12_squared(self, string):
                coeffs, exps = stggtg(self.gamma)
                cf = CorrelationFactor(coeffs, exps)
                baslist = self.get_basis(string)
                b = [i.basisset() for i in baslist]

		#I = np.array(self.mints_.ao_f12_squared(cf,b[0],b[1],b[2],b[3]))
                I = np.array(self.mints_.ao_f12_squared(cf,b[0],b[2],b[1],b[3])).swapaxes(1, 2)
                Imo = self.TransformToMo('f12', I, baslist)

                return Imo


	def f12_double_commutator(self, string):
                coeffs, exps = stggtg(self.gamma)
                cf = CorrelationFactor(coeffs, exps)
                baslist = self.get_basis(string)
                b = [i.basisset() for i in baslist]

                I = I = np.array(self.mints_.ao_f12_double_commutator(cf)).swapaxes(1, 2)
                Imo = self.TransformToMo('f12_double_commutator', I, baslist)

                return Imo	

	

