"""
A Psi4 script to compute MP2-F12 energies.
"""

__authors__   = "Monika Kodrycka"
__credits__ = ["Monika Kodrycka"]

import time
import numpy as np
from helper_mp2f12 import *
np.set_printoptions(precision=5, linewidth=200, threshold=2000, suppress=True)
import psi4


# Memory for Psi4 in GB
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)

# Memory for numpy in GB
numpy_memory = 2

#molecule = psi4.geometry("""
#O
#H 1 R
#H 1 R 2 A

#R = 0.9
#A = 104.5
#symmetry c1
#""")

molecule = psi4.geometry("""
Ne
symmetry c1
""")


psi4.set_options({'basis': 'cc-pvdz',
                  'df_basis_mp2':'cc-pvdz-ri',
		  'scf_type': 'pk',
 		  'mp2_type': 'conv',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8,
		  'FREEZE_CORE': 'True',
                })


mp2f12 = helper_mp2f12(molecule, memory=8)

def Calculate_BC(fk,mp2f12):
        """
        Returns B and C.
        """

	# Build Matrix B
	B = np.zeros((naocc,naocc,naocc,naocc))

	B += mp2f12.f12dc('iiii',1.0)

	tmp = np.einsum('pr,lkqr->klpq', fk[:nobs,:nobs], mp2f12.f12('iipp',1.0))
	B -= np.einsum('mnpq,klpq->klmn', mp2f12.f12('iipp',1.0), tmp)
	tmp = np.einsum('qr,klpr->klpq', fk[:nobs,:nobs], mp2f12.f12('iipp',1.0))
	B -= np.einsum('mnpq,klpq->klmn', mp2f12.f12('iipp',1.0), tmp)
	tmp = np.einsum('px,lkqx->klpq', fk[:nobs,nobs:], mp2f12.f12('iipx',1.0))
	B -= np.einsum('mnpq,klpq->klmn', mp2f12.f12('iipp',1.0), tmp)
	tmp = np.einsum('qx,klpx->klpq', fk[:nobs,nobs:], mp2f12.f12('iipx',1.0))
	B -= np.einsum('mnpq,klpq->klmn', mp2f12.f12('iipp',1.0), tmp)
	tmp = np.einsum('yp,klip->kliy', fk[nobs:,:nobs], mp2f12.f12('iiop',1.0))
	B -= np.einsum('mniy,kliy->klmn', mp2f12.f12('iiox',1.0), tmp)
	tmp = np.einsum('xp,lkjp->klxj', fk[nobs:,:nobs], mp2f12.f12('iiop',1.0))
	B -= np.einsum('nmjx,klxj->klmn', mp2f12.f12('iiox',1.0), tmp)
	tmp = np.einsum('ip,klpy->kliy', fk[:nocc,:nobs], mp2f12.f12('iipx',1.0))
	B -= np.einsum('mniy,kliy->klmn', mp2f12.f12('iiox',1.0), tmp)
	tmp = np.einsum('yx,klix->kliy', fk[nobs:,nobs:], mp2f12.f12('iiox',1.0))
	B -= np.einsum('mniy,kliy->klmn', mp2f12.f12('iiox',1.0), tmp) 
	tmp = np.einsum('xy,lkjy->klxj', fk[nobs:,nobs:], mp2f12.f12('iiox',1.0))
	B -= np.einsum('nmjx,klxj->klmn', mp2f12.f12('iiox',1.0), tmp)
	tmp = np.einsum('jp,lkpx->klxj', fk[:nocc,:nobs], mp2f12.f12('iipx',1.0))
	B -= np.einsum('nmjx,klxj->klmn', mp2f12.f12('iiox',1.0), tmp)
	tmp = np.einsum('ix,klxy->kliy', fk[:nocc,nobs:], mp2f12.f12('iixx',1.0))
	B -= np.einsum('mniy,kliy->klmn', mp2f12.f12('iiox',1.0), tmp)
	tmp = np.einsum('jy,lkyx->klxj', fk[:nocc,nobs:], mp2f12.f12('iixx',1.0))
	B -= np.einsum('nmjx,klxj->klmn', mp2f12.f12('iiox',1.0), tmp)

	# Y contribution 
	tmp = np.einsum('xp,lkbp->klxb', k[nobs:,:nobs], mp2f12.f12('iiap',1.0))
	B -= np.einsum('klxb,nmbx->klmn', tmp, mp2f12.f12('iiax',1.0))
	tmp = np.einsum('yp,klap->klay',k[nobs:,:nobs], mp2f12.f12('iiap',1.0))
	B -= np.einsum('klay,mnay->klmn', tmp, mp2f12.f12('iiax',1.0))
	tmp = np.einsum('xy,lkby->klxb', k[nobs:,nobs:], mp2f12.f12('iiax',1.0))
	B -= np.einsum('klxb,nmbx->klmn', tmp, mp2f12.f12('iiax',1.0))
	tmp = np.einsum('bp,lkpx->klxb', k[nocc:nobs,:nobs], mp2f12.f12('iipx',1.0))
	B -= np.einsum('klxb,nmbx->klmn', tmp, mp2f12.f12('iiax',1.0))
	tmp = np.einsum('ap,klpy->klay', k[nocc:nobs,:nobs], mp2f12.f12('iipx',1.0))
	B -= np.einsum('klay,mnay->klmn', tmp, mp2f12.f12('iiax',1.0))
	tmp = np.einsum('yx,klax->klay',k[nobs:,nobs:], mp2f12.f12('iiax',1.0))
	B -= np.einsum('klay,mnay->klmn', tmp, mp2f12.f12('iiax',1.0))
	tmp = np.einsum('by,lkyx->klxb', k[nocc:nobs,nobs:], mp2f12.f12('iixx',1.0))
	B -= np.einsum('klxb,nmbx->klmn', tmp, mp2f12.f12('iiax',1.0))
	tmp = np.einsum('ax,klxy->klay', k[nocc:nobs,nobs:], mp2f12.f12('iixx',1.0))
	B -= np.einsum('klay,mnay->klmn', tmp, mp2f12.f12('iiax',1.0))
	tmp = np.einsum('xp,klpy->klxy', k[nobs:,:nobs], mp2f12.f12('iipx',1.0))
	B -= np.einsum('klxy,mnxy->klmn', tmp, mp2f12.f12('iixx',1.0))
	tmp = np.einsum('yp,lkpx->klxy', k[nobs:,:nobs], mp2f12.f12('iipx',1.0))
	B -= np.einsum('klxy,mnxy->klmn', tmp, mp2f12.f12('iixx',1.0))
	tmp = np.einsum('yz,lkzx->klxy', k[nobs:,nobs:], mp2f12.f12('iixx',1.0))
	B -= np.einsum('klxy,mnxy->klmn', tmp, mp2f12.f12('iixx',1.0))
	tmp = np.einsum('xz,klzy->klxy', k[nobs:,nobs:], mp2f12.f12('iixx',1.0))
	B -= np.einsum('klxy,mnxy->klmn', tmp, mp2f12.f12('iixx',1.0))

	# F^2 Contribution
	B += np.einsum('kp,nmlp->klmn', fk[nfocc:nocc,:nobs], mp2f12.f12squared('iiip',1.0))
	B += np.einsum('lp,mnkp->klmn', fk[nfocc:nocc,:nobs], mp2f12.f12squared('iiip',1.0))
	B += np.einsum('kx,nmlx->klmn', fk[nfocc:nocc,nobs:], mp2f12.f12squared('iiix',1.0))
	B += np.einsum('lx,mnkx->klmn', fk[nfocc:nocc,nobs:], mp2f12.f12squared('iiix',1.0))

	# C Matrix
        C = np.zeros((naocc,naocc,nvir,nvir))
        C += np.einsum('bx,klax->klab', f[nocc:nobs,nobs:], mp2f12.f12('iiax',1.0))
        C += np.einsum('ax,lkbx->klab', f[nocc:nobs,nobs:], mp2f12.f12('iiax',1.0))

        # -FC contribution to B
        B -= np.einsum('klab,mnab->klmn', mp2f12.f12('iiaa',1.0), C)

	# Symmetrize B
	B = 0.5 * (B + np.einsum('klmn->mnkl', B))

	return B, C


def compute_V_tilde_so_(mp2f12, i, j, f, V, C):
	"""
    	Returns V_tilde_so.
   	"""
    	obs = mp2f12.get_size()
	nocc = obs['o']
	naocc = obs['i']
	nfocc = nocc - naocc
	nvir = obs['a']

    	V_ab = [[]]
    	V_aa = [[]]
    	G_oovv = mp2f12.g('iiaa')
   	for k in range(naocc):
        	for l in range(naocc):
            		V_ab_kl = V[k,l,i,j]
            		V_aa_kl = V[k,l,i,j] - V[l,k,i,j]
            		for a in range(nvir):
                		for b in range(nvir):
                    			D_ijab = f[nocc+a,nocc+a] + f[nocc+b,nocc+b] \
                           			- f[nfocc+i,nfocc+i] - f[nfocc+j,nfocc+j]
                    			C_klab = C[k,l,a,b]
                    			C_klba = C[k,l,b,a]
                    			G_ijab = G_oovv[i,j,a,b]
                    			G_ijba = G_oovv[i,j,b,a]
                    			V_ab_kl -= C_klab * G_ijab / D_ijab
                    			if k < l and a < b:
                        			C_klab = C_klab - C_klba
                        			G_ijab = G_ijab - G_ijba
                       				V_aa_kl -= C_klab * G_ijab / D_ijab
            		V_ab[-1].append(V_ab_kl)
            		if k < l:
                		V_aa[-1].append(V_aa_kl)

    	return np.matrix(V_ab).T, np.matrix(V_aa).T


def compute_B_tilde_so_(mp2f12, i, j, f, B, X, C):
	"""
    	Returns B_tilde_so.
    	"""

        obs = mp2f12.get_size()
        nocc = obs['o']
        naocc = obs['i']
        nfocc = nocc - naocc
        nvir = obs['a']


    	e_ij = f[nfocc+i,nfocc+i] + f[nfocc+j,nfocc+j]
    	B_aa = []
    	B_ab = []
    	for k in range(naocc):
        	for l in range(naocc):
            		B_ab.append([])
            		if k < l: B_aa.append([])
            		B_ab_klmn = 0.0
           		B_aa_klmn = 0.0
            		for m in range(naocc):
                		for n in range(naocc):
                    			B_ab_klmn = B[k,l,m,n] - e_ij * X[k,l,m,n]
                    			if k < l and m < n:
                        			B_aa_klmn = B[k,l,m,n] - B[k,l,n,m] \
                            			- e_ij * X[k,l,m,n] + e_ij * X[k,l,n,m]
                    			for a in range(nvir):
                        			for b in range(nvir):
                            				D_ijab = f[nocc+a,nocc+a] + f[nocc+b,nocc+b] \
                                   			- f[nfocc+i,nfocc+i] - f[nfocc+j,nfocc+j]
                            				C_klab = C[k,l,a,b]
                            				C_klba = C[k,l,b,a]
                            				C_mnab = C[m,n,a,b]
                            				C_mnba = C[m,n,b,a]
                            				B_ab_klmn -= C_klab * C_mnab / D_ijab
                            				if a < b:
                                				C_klab = C_klab - C_klba
                                				C_mnab = C_mnab - C_mnba
                               					B_aa_klmn -= C_klab * C_mnab / D_ijab
                    			B_ab[-1].append(B_ab_klmn)
                    			if k < l and m < n:
                        			B_aa[-1].append(B_aa_klmn)

    	return np.matrix(B_ab), np.matrix(B_aa)


print('\n\n********************* MP2-F12(3C) ***********************')
print('[Werner, Adler, Manby, J. Chem. Phys. 2007, 126, 164102.]\n\n')

obs = mp2f12.get_size()
nocc = obs['o']
naocc = obs['i']
nfocc = nocc - naocc 
nvir = obs['a']
nobs = nmo = obs['p']
ncabs = obs['x']
nri = nmo + ncabs
eps = mp2f12.get_eps()
Eocc = eps[nfocc:nocc]
Evirt = eps[nocc:]

#constant
psi_hartree2kcalmol = 627.5095

# Orbital spaces
print('\nOrbital Spces:\n')
print('  nfocc: %d' % nfocc)
print('  naocc: %d' % naocc)
print('   nocc: %d' % nocc)
print('   nvir: %d' % nvir)
print('   nobs: %d' % nobs)
print('  ncabs: %d\n' % ncabs)


print "\nStart calculations ...\n\n"


# Build Marrix V
V = np.zeros((naocc,naocc,naocc,naocc))
V += mp2f12.f12g12('iiii', 1.0)
V -= np.einsum('ijrs,klrs->ijkl',mp2f12.g('iipp'),mp2f12.f12('iipp',1.0))
V -= np.einsum('ijxm,klxm->ijkl',mp2f12.g('iixi'),mp2f12.f12('iixi',1.0))
V -= np.einsum('ijmx,klmx->ijkl',mp2f12.g('iiix'),mp2f12.f12('iiix',1.0))


#Build Matrix X
X = np.zeros((naocc,naocc,naocc,naocc))
X += mp2f12.f12squared('iiii', 1.0)
X -= np.einsum('ijrs,klrs->ijkl',mp2f12.f12('iipp',1.0),mp2f12.f12('iipp',1.0))
X -= np.einsum('ijxm,klxm->ijkl',mp2f12.f12('iixi',1.0),mp2f12.f12('iixi',1.0))
X -= np.einsum('ijmx,klmx->ijkl',mp2f12.f12('iiix',1.0),mp2f12.f12('iiix',1.0))


V_pp, V_px , V_xx = mp2f12.get_V()
T_pp, T_px , T_xx = mp2f12.get_T()


# Build Fock matrix
k = np.zeros((nri,nri))
f = np.zeros((nri,nri))

# T1 and V1 contribution to the Fock matrix
f[:nobs,:nobs] = T_pp + V_pp
f[:nobs,nobs:] = T_px + V_px
f[nobs:,nobs:] = T_xx + V_xx

# Coulomb integral contribution to the Fock matrix
f[:nobs,:nobs] += 2.0 * np.einsum('viui->vu', mp2f12.g('popo'))
f[:nobs,nobs:] += 2.0 * np.einsum('iviu->vu', mp2f12.g('opox'))
f[nobs:,nobs:] += 2.0 * np.einsum('iviu->vu', mp2f12.g('oxox'))

# Exchange integral contribution to the Fock matrix
k[:nobs,:nobs] = np.einsum('iivu->vu', mp2f12.g('oopp'))
k[:nobs,nobs:] = np.einsum('iivu->vu', mp2f12.g('oopx'))
k[nobs:,nobs:] = np.einsum('iivu->vu', mp2f12.g('ooxx'))
f -= k

# Fill in the remaining elements by symmetry
f[nobs:,:nobs] = f[:nobs,nobs:].T
k[nobs:,:nobs] = k[:nobs,nobs:].T
fk = f + k

B,C = Calculate_BC(fk, mp2f12)


G_oovv = mp2f12.g('iiaa')

# Conventional MP2
E_mp2 = 0.0
e_aa_pairs = {}
e_ab_pairs = {}
for i in range(naocc):
        e_aa_pairs[i] = {}
        e_ab_pairs[i] = {}
        for j in range(naocc):
                e_ab_pairs[i][j] = 0.0
                e_aa_pairs[i][j] = 0.0
                for a in range(nvir):
                        for b in range(nvir):
                                G_ijab = G_oovv[i,j,a,b]
                                G_ijba = G_oovv[i,j,b,a]
                                D_ijab = f[nocc+a,nocc+a] + f[nocc+b,nocc+b] \
                                	- f[nfocc+i,nfocc+i] - f[nfocc+j,nfocc+j]
                                e_ab_pairs[i][j] -= G_ijab * G_ijab / D_ijab
                                if i > j or a > b: continue
                                G_ijab = G_ijab - G_ijba
                                e_aa_pairs[i][j] -= G_ijab * G_ijab / D_ijab
                E_mp2 += e_ab_pairs[i][j] + 2.0 * e_aa_pairs[i][j]



# F12 contribution
E_f12 = 0.0
f_aa_pairs = {}
f_ab_pairs = {}
for i in range(naocc):
	f_aa_pairs[i] = {}
       	f_ab_pairs[i] = {}
        for j in range(naocc):
        	f_ab_pairs[i][j] = 0.0
            	f_aa_pairs[i][j] = 0.0
            	V_ab, V_aa = compute_V_tilde_so_(mp2f12, i, j, f, V, C)
            	B_ab, B_aa = compute_B_tilde_so_(mp2f12, i, j, f, B, X, C)
            	B_ab_inv = B_ab.I
            	f_ab_pairs[i][j] = - (V_ab.T * B_ab_inv * V_ab)[0,0]
            	if i < j:
                	B_aa_inv = B_aa.I
                	f_aa_pairs[i][j] = - (V_aa.T * B_aa_inv * V_aa)[0,0]
            	E_f12 += f_ab_pairs[i][j] + 2.0 * f_aa_pairs[i][j]


# Printing the Alpha-Beta pair energies
print('\nAlpha-Beta pair energies:\n')
print('  %5s %5s %16s %16s %16s\n' % ('i', 'j', 'mp2', 'f12', 'total'))
for i in range(naocc):
	for j in range(naocc):
        	e_mp2 = e_ab_pairs[i][j]
            	e_f12 = f_ab_pairs[i][j]
            	e = e_mp2 + e_f12
            	print('  %5d %5d %16.9f %16.9f %16.9f\n' % \
        		(i+1, j+1, e_mp2, e_f12, e))


# Printing the Alpha-Alpha pair energies
print('\nAlpha-Alpha pair energies:\n')
print('  %5s %5s %16s %16s %16s\n' % (' i', 'j', 'mp2', 'f12', 'total'))
for i in range(naocc):
	for j in range(0, i, 1):
        	e_mp2 = e_aa_pairs[j][i]
            	e_f12 = f_aa_pairs[j][i]
            	e = e_mp2 + e_f12
            	print('  %5d %5d %16.9f %16.9f %16.9f\n' % \
                (i+1, j+1, e_mp2, e_f12, e))


# Obtain SCF from Psi4
E_scf = psi4.energy('SCF', return_wfn=False)

print('\nMP2-F12 Energy:')
print('----------------------------------')
print('                  SCF energy: %16.9f' % (E_scf))
print('      MP2 correlation energy: %16.9f'   % (E_mp2))
print('      F12 correlation energy: %16.9f'   % (E_f12))
print('  MP2-F12 correlation energy: %16.9f'   % (E_mp2 + E_f12))
print('        MP2-F12 total energy: %16.9f'   % (E_scf + E_mp2 + E_f12))
