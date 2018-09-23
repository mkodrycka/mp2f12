"""
Build CABS space
"""

import scipy.linalg 
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4


def make_ortho(S, ThrAbs, ThrRel):
        evals, evecs = np.linalg.eigh(S)
        ndim = len(evals)
        maxeval = max(evals)
        sigma =np.zeros(S.shape)
        threshold = max(ThrAbs, maxeval*ThrRel)

        ndel = 0
        for i in range(ndim):
                if abs(evals[i]) < threshold: ndel += 1
                else: sigma[i,i] = 1.0 / np.sqrt(evals[i])

        nDimOrtho =  ndim - ndel
        sigma = sigma[ndel:,ndel:]
        evecs = evecs[:,ndel:]

        OrthoBasis = np.einsum('ij,jk->ik',evecs,sigma)

        return OrthoBasis, nDimOrtho


def get_cabs(conv, aux_basis, wfn):
	# Constants used in the program
	ThrAbs = 1.00E-06
	ThrRel = 1.00E-08

	# Build Abs Vector
	mints = psi4.core.MintsHelper(wfn.basisset())
	Saa = np.array(mints.ao_overlap(aux_basis, aux_basis))
	ABS, nabs = make_ortho(Saa, ThrAbs, ThrRel)

	# Build CABS vector
	Cobs = wfn.Ca()
	SAoRi = mints.ao_overlap(conv,aux_basis) 
	SMoAbs = np.einsum("pq,pP->Pq", SAoRi, Cobs)
	SMoAbs = np.einsum("Pq,qQ->PQ", SMoAbs, ABS)

	# Build Sstar
	Sstar = np.identity(nabs)
	Sstar -= np.einsum("ij,jk->ik",SMoAbs.T,SMoAbs)
	# Build Cstar
	Cstar, ncabs = make_ortho(Sstar, ThrAbs, ThrRel)

	# Finaly we need to transform our new CABS vectors back into AO/RI basis
	Crx = -1.0*np.einsum("ij,jk->ik",SMoAbs,Cstar)
	cAO_CABS = np.einsum("ij,jk->ik",Cobs,Crx)
	cAUX_CABS = np.einsum("ij,jk->ik",ABS,Cstar)

	return cAO_CABS, cAUX_CABS

