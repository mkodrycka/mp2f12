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
 		  'mp2_type': 'conv',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8,
		  'FREEZE_CORE': 'True',
                })


mp2f12 = helper_mp2f12(molecule, memory=8)

def Calculate_B(fk, mp2f12):

	obs = mp2f12.get_size()
	nocc = obs['o']
	naocc = obs['i']
	nfocc = nocc - naocc
	nvir = obs['a']
	nobs = nmo = obs['p']
	ncabs = obs['x']
	nri = nmo + ncabs
	eps = mp2f12.get_eps()

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

	return B


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


print "\n Start MP2-F12 with fixed amplitudes...\n\n"

# Traditional MP2
D = Eocc.reshape(-1, 1, 1, 1) + Eocc.reshape(-1, 1, 1) - Evirt.reshape(-1, 1) - Evirt
t_ijab = mp2f12.g('iiaa') / D 
T_ijab = 2 * t_ijab - t_ijab.swapaxes(2,3)
e_mp2 = np.einsum('ijab,ijab->', mp2f12.g('iiaa'), T_ijab)


# Build Marrix V
V = np.zeros((naocc,naocc,naocc,naocc))
V += mp2f12.f12g12('iiii', 1.0)
V -= np.einsum('ijrs,klrs->ijkl',mp2f12.g('iipp'),mp2f12.f12('iipp',1.0))
V -= np.einsum('ijxm,klxm->ijkl',mp2f12.g('iixi'),mp2f12.f12('iixi',1.0))
V -= np.einsum('ijmx,klmx->ijkl',mp2f12.g('iiix'),mp2f12.f12('iiix',1.0))


e_f12_V = 0.0
for i in range(naocc):
        for j in range(naocc):
            e_f12_V += (5.0/8.0) * V[i,j,i,j]
            e_f12_V -= (1.0/8.0) * V[j,i,i,j]


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


e_f12_X = 0.0
for i in range(naocc):
        for j in range(naocc):
                #e_f12_X += (f[i,i] + f[j,j]) * (7.0/32.0) * X[i,j,i,j]
		#e_f12_X += (f[i,i] + f[j,j]) * (1.0/32.0) * X[i,j,j,i]
		e_f12_X += (f[nfocc+i,nfocc+i] + f[nfocc+j,nfocc+j]) * (7.0/32.0) * X[i,j,i,j]
		e_f12_X += (f[nfocc+i,nfocc+i] + f[nfocc+j,nfocc+j]) * (1.0/32.0) * X[i,j,j,i]

	

B = Calculate_B(fk, mp2f12)

e_f12_B = 0.0
for i in range(naocc):
        for j in range(naocc):
                e_f12_B += (7.0/32.0) * B[i,j,i,j]
                e_f12_B += (1.0/32.0) * B[i,j,j,i]


# Calculate F12 correction
e_f12 = 2*e_f12_V + e_f12_B - e_f12_X

scf_e = psi4.energy('SCF', return_wfn=False)
mp2_e = psi4.energy('MP2')

print('\nMP2-F12 with fixed amplitudes:\n')
print('      MP2 correlation energy: %16.9f' % (e_mp2))
print('      F12 correlation energy: %16.9f' % (e_f12))
print('  MP2-F12 correlation energy: %16.9f' % (e_mp2 + e_f12))
print(' MP2-F12 total energy: %16.9f' % (scf_e + e_mp2 + e_f12))




