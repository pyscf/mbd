#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Jan Hermann
#

'''
Many-Body van der Waals Interactions

Refs:
    Phys. Rev. Lett. 108, 236402
'''

import numpy as np
import scipy.special
from .vdw_param import vdw_params

def get_dipole(version, coords, alpha, R_vdw, beta, a):
    sigma = ((2./np.pi)**.5 * alpha/3) ** (2./3)
    sigma_ij = (sigma[:,None] + sigma) ** .5
    R_vdw_ij = R_vdw[:,None] + R_vdw
    r = coords[:,None,:] - coords
    if shift is not None:
        r -= shift
    rr = r[:,None,:,None] + r[:,None,:]
    r2 = np.einsum('ijxx->ij', rr)

    if version.upper() == 'BARE':
        r5 = r2 ** 2.5
        dip = (np.einsum('ij,xy->ixjy', r2, np.eye(3)) - 3 * rr) / r5[:,None,:,None]

    elif version.upper() == 'FERMI,DIP,GG':
        r_norm = r2 ** .5
        r5 = r_norm ** 5
        dip = (r2 * np.eye(3) - 3 * rr) / r5

        r_sigma = r_norm / sigma
        a1 = -2. / np.pi**.5 * r_sigma * np.exp(-r_sigma**2)
        zeta1 = scipy.special.erf(r_sigma) + a1
        zeta2 = a1 * (2 * r_sigma**2)
        dip = dip * zeta1[:,None,:,None] - (zeta2 / r5)[:,None,:,None] * rr

        damping = 1. - 1. / (1. + np.exp(-a * (r_norm / (beta*R_vdw_ij) - 1.)))
        dip *= damping[:,None,:,None]

    elif version.upper() == 'FERMI,DIP':
        r_norm = r2 ** .5
        r5 = r_norm ** 5
        dip = (np.einsum('ij,xy->ixjy', r2, np.eye(3)) - 3 * rr) / r5[:,None,:,None]
        damping = 1. / (1. + np.exp(-a * (r_norm / (beta*R_vdw_ij) - 1.)))
        dip *= damping[:,None,:,None]

    else:
        raise RuntimeError

    n = coords.shape[0]
    return dip.reshape(3*n, 3*n)


def get_freq_grid(n, L=0.6):
    x, w = np.polynomial.legendre.leggauss(n)
    w = 2*L/(1-x)**2*w
    x = L*(1+x)/(1-x)
    return np.hstack([[0], x]), np.hstack([[0], w])


def get_omega(C6, alpha_0):
    return 4./3*C6/alpha_0**2


def mbd_rsscs(atoms, volumes, beta, ngrid=15):
    species = [a[0] for a in atoms]
    coords = np.array([a[1] for a in atoms], dtype=np.float64)
    volumes = np.array(volumes, dtype=np.float64)
    alpha_0_free, C6_free, R_vdw_free = [
        np.array([vdw_params[s][quantity] for s in species])
        for quantity in 'alpha_0 C6 R_vdw'.split()
    ]
    alpha_0, C6, R_vdw = \
        alpha_0_free*volumes, C6_free*volumes**2, R_vdw_free*volumes**(1./3)
    omega = get_omega(C6, alpha_0)
    freqs, freqs_w = get_freq_grid(ngrid)
    alpha_scs_dyn = []
    for freq, freq_w in zip(freqs, freqs_w):
        alpha = alpha_0/(1+(freq/omega)**2)
        T = get_dipole('fermi,dip,gg', coords, alpha=alpha, R_vdw=R_vdw, beta=beta, a=6)
        alpha_scs = np.linalg.inv(np.diag(1/alpha.repeat(3))+T)
        alpha_scs_dyn.append([
            sum(alpha_scs[j::3, 3*i+j].sum() for j in range(3))/3
            for i in range(len(atoms))
        ])
    alpha_scs_dyn = np.array(alpha_scs_dyn)
    alpha_0, C6, R_vdw = \
        alpha_scs_dyn[0], \
        3./np.pi*np.sum(alpha_scs_dyn**2*freqs_w[:, None], 0), \
        R_vdw*(alpha_scs_dyn[0]/alpha_0)**(1./3)
    omega = get_omega(C6, alpha_0)
    T = get_dipole('fermi,dip', coords, R_vdw=R_vdw, beta=beta, a=6)
    C = np.diag((omega**2).repeat(3)) + \
        np.sqrt(alpha_0.repeat(3)[:, None]*alpha_0.repeat(3)[None, :]) * \
        (omega.repeat(3)[:, None]*omega.repeat(3)[None, :]) * T
    eigs = np.linalg.eigvalsh(C)
    ene = np.sqrt(eigs).sum()/2 - 3./2*omega.sum()
    return ene
