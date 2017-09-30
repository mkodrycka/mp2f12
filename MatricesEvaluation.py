"""
Matrices evaluation based on the Werner, Adler, Manby, J. Chem. Phys. 2007, 126, 164102 papier.
"""

from integrals_helper import Integrals
import numpy as np


def Compute_BC(bs, correlation_factor):

	nfocc = bs['focc'].dim()[0]
	naocc = bs['aocc'].dim()[0]
	nocc = bs['occ'].dim()[0]
	nvir = bs['vir'].dim()[0]
	nobs = bs['obs'].dim()[0]
	ncabs = bs['cabs'].dim()[0]
 	nri = nobs + ncabs

    	# Empty integral objects
    	integrals = Integrals(bs, correlation_factor)

    	# Zero arrays for V, X, B, C
    	B = np.zeros((naocc,naocc,naocc,naocc))
    	C = np.zeros((naocc,naocc,nvir,nvir))

    	# 'Diagonal' contributions to V and B
    	B += integrals.f12_double_commutator('aaaa')

    	# Fock matrix
    	k = np.zeros((nri,nri))
    	f = np.zeros((nri,nri))

    	# T1 and V1 contribution to the Fock matrix
    	f[:nobs,:nobs] = integrals.t('pp') + integrals.v('pp')
    	f[:nobs,nobs:] = integrals.t('px') + integrals.v('px')
    	f[nobs:,nobs:] = integrals.t('xx') + integrals.v('xx')
    	h_px = integrals.t('px') +  integrals.v('px')

    	# Coulomb integral contribution to the Fock matrix
    	f[:nobs,:nobs] += 2.0 * np.einsum('viui->vu', integrals.g('popo'))
    	f[:nobs,nobs:] += 2.0 * np.einsum('iviu->vu', integrals.g('opox'))
    	f[nobs:,nobs:] += 2.0 * np.einsum('iviu->vu', integrals.g('oxox'))

    	# Exchange integral contribution to the Fock matrix
    	k[:nobs,:nobs] = np.einsum('iivu->vu', integrals.g('oopp'))
    	k[:nobs,nobs:] = np.einsum('iivu->vu', integrals.g('oopx'))
    	k[nobs:,:nobs] = np.einsum('px->xp', k[:nobs,nobs:])
    	k[nobs:,nobs:] = np.einsum('iivu->vu', integrals.g('ooxx'))
    	k[nobs:,nobs:] = np.einsum('iivu->vu', integrals.g('ooxx'))
    	f -= k

    	# Fill in the remaining elements by symmetry
    	f[nobs:,:nobs] = f[:nobs,nobs:].T
    	k[nobs:,:nobs] = k[:nobs,nobs:].T
    	fk = f + k


    	tmp = np.einsum('pr,lkqr->klpq', fk[:nobs,:nobs], integrals.f12('aapp'))
    	B -= np.einsum('mnpq,klpq->klmn', integrals.f12('aapp'), tmp)
    	tmp = np.einsum('qr,klpr->klpq', fk[:nobs,:nobs], integrals.f12('aapp'))
    	B -= np.einsum('mnpq,klpq->klmn', integrals.f12('aapp'), tmp)
    	tmp = np.einsum('px,lkqx->klpq', fk[:nobs,nobs:], integrals.f12('aapx'))
    	B -= np.einsum('mnpq,klpq->klmn', integrals.f12('aapp'), tmp)
    	tmp = np.einsum('qx,klpx->klpq', fk[:nobs,nobs:], integrals.f12('aapx'))
    	B -= np.einsum('mnpq,klpq->klmn', integrals.f12('aapp'), tmp)
    	tmp = np.einsum('yp,klip->kliy', fk[nobs:,:nobs], integrals.f12('aaop'))
    	B -= np.einsum('mniy,kliy->klmn', integrals.f12('aaox'), tmp)
    	tmp = np.einsum('xp,lkjp->klxj', fk[nobs:,:nobs], integrals.f12('aaop'))
    	B -= np.einsum('nmjx,klxj->klmn', integrals.f12('aaox'), tmp)
    	tmp = np.einsum('ip,klpy->kliy', fk[:nocc,:nobs], integrals.f12('aapx'))
    	B -= np.einsum('mniy,kliy->klmn', integrals.f12('aaox'), tmp)
    	tmp = np.einsum('yx,klix->kliy', fk[nobs:,nobs:], integrals.f12('aaox'))
    	B -= np.einsum('mniy,kliy->klmn', integrals.f12('aaox'), tmp) 
    	tmp = np.einsum('xy,lkjy->klxj', fk[nobs:,nobs:], integrals.f12('aaox'))
    	B -= np.einsum('nmjx,klxj->klmn', integrals.f12('aaox'), tmp)
    	tmp = np.einsum('jp,lkpx->klxj', fk[:nocc,:nobs], integrals.f12('aapx'))
    	B -= np.einsum('nmjx,klxj->klmn', integrals.f12('aaox'), tmp)
    	tmp = np.einsum('ix,klxy->kliy', fk[:nocc,nobs:], integrals.f12('aaxx'))
    	B -= np.einsum('mniy,kliy->klmn', integrals.f12('aaox'), tmp)
    	tmp = np.einsum('jy,lkyx->klxj', fk[:nocc,nobs:], integrals.f12('aaxx'))
    	B -= np.einsum('nmjx,klxj->klmn', integrals.f12('aaox'), tmp)

 
    	#-Y contribution to B
    	tmp = np.einsum('xp,lkbp->klxb', k[nobs:,:nobs], integrals.f12('aapp')[:,:,nocc:,:])
    	B -= np.einsum('klxb,nmbx->klmn', tmp, integrals.f12('aapx')[:,:,nocc:,:])
    	tmp = np.einsum('yp,klap->klay',k[nobs:,:nobs], integrals.f12('aapp')[:,:,nocc:,:])
    	B -= np.einsum('klay,mnay->klmn', tmp, integrals.f12('aapx')[:,:,nocc:,:])
    	tmp = np.einsum('xy,lkby->klxb', k[nobs:,nobs:], integrals.f12('aapx')[:,:,nocc:,:])
    	B -= np.einsum('klxb,nmbx->klmn', tmp, integrals.f12('aapx')[:,:,nocc:,:])
    	tmp = np.einsum('bp,lkpx->klxb', k[nocc:nobs,:nobs], integrals.f12('aapx'))
    	B -= np.einsum('klxb,nmbx->klmn', tmp, integrals.f12('aapx')[:,:,nocc:,:])
    	tmp = np.einsum('ap,klpy->klay', k[nocc:nobs,:nobs], integrals.f12('aapx'))
    	B -= np.einsum('klay,mnay->klmn', tmp, integrals.f12('aapx')[:,:,nocc:,:])
    	tmp = np.einsum('yx,klax->klay',k[nobs:,nobs:], integrals.f12('aapx')[:,:,nocc:,:])
    	B -= np.einsum('klay,mnay->klmn', tmp, integrals.f12('aapx')[:,:,nocc:,:])
    	tmp = np.einsum('by,lkyx->klxb', k[nocc:nobs,nobs:], integrals.f12('aaxx'))
    	B -= np.einsum('klxb,nmbx->klmn', tmp, integrals.f12('aapx')[:,:,nocc:,:])
    	tmp = np.einsum('ax,klxy->klay', k[nocc:nobs,nobs:], integrals.f12('aaxx'))
        B -= np.einsum('klay,mnay->klmn', tmp, integrals.f12('aapx')[:,:,nocc:,:])
        tmp = np.einsum('xp,klpy->klxy', k[nobs:,:nobs], integrals.f12('aapx'))
        B -= np.einsum('klxy,mnxy->klmn', tmp, integrals.f12('aaxx'))
    	tmp = np.einsum('yp,lkpx->klxy', k[nobs:,:nobs], integrals.f12('aapx'))
    	B -= np.einsum('klxy,mnxy->klmn', tmp, integrals.f12('aaxx'))
    	tmp = np.einsum('yz,lkzx->klxy', k[nobs:,nobs:], integrals.f12('aaxx'))
    	B -= np.einsum('klxy,mnxy->klmn', tmp, integrals.f12('aaxx'))
    	tmp = np.einsum('xz,klzy->klxy', k[nobs:,nobs:], integrals.f12('aaxx'))
    	B -= np.einsum('klxy,mnxy->klmn', tmp, integrals.f12('aaxx'))

    	# F^2 Contribution
    	B += np.einsum('kp,nmlp->klmn', fk[nfocc:nocc,:nobs], integrals.f12_squared('aaap'))
    	B += np.einsum('lp,mnkp->klmn', fk[nfocc:nocc,:nobs], integrals.f12_squared('aaap'))

    	# F2(ooox)
    	B += np.einsum('kx,nmlx->klmn', fk[nfocc:nocc,nobs:], integrals.f12_squared('aaax'))
    	B += np.einsum('lx,mnkx->klmn', fk[nfocc:nocc,nobs:], integrals.f12_squared('aaax'))


	# Symmetrize B
	B = 0.5 * (B + np.einsum('klmn->mnkl', B))


	return f, B, C


def Compute_V(bs, correlation_factor):
	"""
	Return a numpy array V_{mn}^{ij} = <ij|r_{12}^{-1}Q_{12}F_{12}|mn>
    	"""
	
	integrals = Integrals(bs, correlation_factor)

	# Unpack variables needed from bs
	naocc = bs['aocc'].dim()[0]

	# Allocate space for V
	V = np.zeros((naocc,naocc,naocc,naocc))

	# FG contribution
 	V += integrals.f12g12('aaaa')

	# F(oopp) contribution
    	V -= np.einsum('klrs,ijrs->ijkl', integrals.g('aapp'), integrals.f12('aapp'))

    	# F(oopx) contribution
    	V -= np.einsum('ijmy,klmy->klij', integrals.g('aaox'), integrals.f12('aaox'))
    	V -= np.einsum('jinx,lknx->klij', integrals.g('aaox'), integrals.f12('aaox'))

    	return V

def Compute_X(bs, correlation_factor):
   	""" 
	Return a numpy array X_{mn}^{ij} = <ij|r_{12}^{-1}Q_{12}F_{12}|mn>
	"""    

	integrals = Integrals(bs, correlation_factor)

    	# Unpack variables needed from bs
    	naocc = bs['aocc'].dim()[0]

    	# Allocate space for V
    	X = np.zeros((naocc,naocc,naocc,naocc))

    	# FG contribution
    	X += integrals.f12_squared('aaaa')

    	# F(oopp) contribution
    	X -= np.einsum('klrs,ijrs->ijkl', integrals.f12('aapp'), integrals.f12('aapp'))

    	# F(oopx) contribution
    	X -= np.einsum('ijmy,klmy->klij', integrals.f12('aaox'), integrals.f12('aaox'))
    	X -= np.einsum('jinx,lknx->klij', integrals.f12('aaox'), integrals.f12('aaox'))

    	return X

