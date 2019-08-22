import math
import numpy as np
from pwinput import PWInput

# Partly adapted from https://github.com/attacc/QEplayground/blob/master/QEplayground/supercell.py

def find_integers(m1, m2, m3, g23, g12, g31, g123):
    """
    Compute integers for off-diagonal supercell matrix elements
    Called by find_nondiagonal()
    """
    # Compute p (it's a modulo equation)
    if g23 == 1:
        p = 0
    else:
        for i in range(0, g23):
            if (m2 + i * m3) % g23 == 0:
                p = i
                break
    # Compute q
    g12_r = g12 // g123
    g23_r = g23 // g123
    g31_r = g31 // g123
    if g12_r == 1:
        q = 0
    else:
        for i in range(0, g12_r):
            if (g23_r * m1 + i * g31_r * m2) % g12_r == 0:
                q = i
                break
    # Compute r
    gg_r = g31 * g23 // g123
    z = (g23 * m1 + g31 * q * m2) // g12
    if gg_r == 1:
        r = 0
    else:
        for i in range(0, gg_r):
            if (z + i * m3) % gg_r == 0:
                r = i
                break
    return p, q, r

def find_nondiagonal(mlist, nlist):
    """
    Nondiagonal supercell, based on [Phys. Rev. B 92, 184301]
    """
    for i in range(3):
        # Take care of components already at Gamma
        if mlist[i] == 0:
            nlist[i] = 1
        # Shift the q-point into the positive quadrant of the reciprocal unit cell
        mlist[i] = 1 + (mlist[i] - 1) % nlist[i]
    m1, m2, m3 = mlist
    n1, n2, n3 = nlist
    # GCDs of n1, n2, n3 (in the logical order of the derivation)
    g23 = math.gcd(n2, n3)
    g12 = math.gcd(n1, n2)
    g31 = math.gcd(n3, n1)
    g123 = math.gcd(n1, math.gcd(n2, n3))
    # Integers needed to solve the supercell matrix equation
    p, q, r = find_integers(m1, m2, m3, g23, g12, g31, g123)
    # Matrix elements (in order of derivation) and supercell matrix
    S_33 = n3
    S_22 = n2 // g23
    S_23 = p * n3 // g23
    S_11 = g123 * n1 // (g12 * g31)
    S_12 = q * g123 * n2 // (g12 * g23)
    S_13 = r * g123 * n3 // (g31 * g23)
    S = np.array([[S_11, S_12, S_13], [0, S_22, S_23], [0, 0, S_33]])
    return S

def reduce_vec(vecs):
    """
    Adapted from the supplementary matrerial of PHYSICAL REVIEW B 92, 184301

    Given three linearly independent input vectors a, b and c, construct the
    following linear combinations: a+b-c, a-b+c, -a+b+c, a+b+c and  check if any
    of the four new vectors is shorter than any of a, b or c. If so, replace the
    longest of a, b and c with the new (shorter) vector. The resulting three
    vectors are also linearly independent.
    """
    tol_zero = 1E-7

    # Determine which of the three input vectors is the longest
    maxlen = 0
    for i in range(3):
        nlen = np.sum(vecs[:, i]**2)
        # Test nlen>maxlen within some tolerance to avoid floating point problems
        if nlen - maxlen > tol_zero * maxlen:
            maxlen = nlen
            longest = i

    # Construct the four linear combinations
    newvecs = np.zeros((3, 4))
    newvecs[:, 0] =  vecs[:, 0] + vecs[:, 1] - vecs[:, 2]
    newvecs[:, 1] =  vecs[:, 0] - vecs[:, 1] + vecs[:, 2]
    newvecs[:, 2] = -vecs[:, 0] + vecs[:, 1] + vecs[:, 2]
    newvecs[:, 3] =  vecs[:, 0] + vecs[:, 1] + vecs[:, 2]

    # Check if any of the four new vectors is shorter than longest input vector
    reduce_vec = False
    for i in range(4):
        nlen = np.sum(newvecs[:,i]**2)
        if nlen - maxlen < - tol_zero * maxlen:
            vecs[:, longest] = newvecs[:, i]
            reduce_vec = True
            break

    return reduce_vec

def minkowski_reduce(vecs_in):
    """
    Adapted from the supplementary matrerial of PHYSICAL REVIEW B 92, 184301

    Given n vectors a(i) that form a basis for a lattice L in n dimensions, the
    a(i) are said to be Minkowski-reduced if the following conditions are met:

    - a(1) is the shortest non-zero vector in L
    - for i>1, a(i) is the shortest possible vector in L such that a(i)>=a(i-1)
      and the set of vectors a(1) to a(i) are linearly independent

    In other words the a(i) are the shortest possible basis vectors for L. This
    routine, given a set of input vectors a'(i) that are possibly not
    Minkowski-reduced, returns the vectors a(i) that are.
    """
    vecs = np.copy(vecs_in)
    while True:
        tempvec = np.copy(vecs)
        for i in range(3):
            # First check linear combinations involving two vectors
            vecs[:, i] = 0.0
            changed = reduce_vec(vecs)
            vecs[:, i] = np.copy(tempvec[:, i])
            if changed:
                break
        if changed:
            continue
        # Then check linear combinations involving all three
        if reduce_vec(vecs):
            continue
        break

    # Change order of vecs to make vecs[:, 0] shortest and vecs[:, 2] longest
    ind = np.argsort(np.sum(vecs**2, axis=0))
    vecs = vecs[:, ind]

    return vecs

def atom_supercell(pw: PWInput, s_mat):
    # Return new atomic position in bohr
    from itertools import product

    if s_mat[1, 0] != 0 or s_mat[2, 0] != 0 or s_mat[2, 1] != 0:
        raise ValueError("s_mat must be an upper-triangular matrix")
    if not isinstance(s_mat[0,0], np.integer):
        raise ValueError("s_mat must be an integer matrix")

    s_diag = np.diag(s_mat)
    multiplier = s_diag[0] * s_diag[1] * s_diag[2]
    cell = pw.cell
    atoms_cart = pw.atoms_cart
    atoms_cart_new = np.empty((3, pw.nat * multiplier), dtype=float)

    atoms_name_new = pw.atoms_name * multiplier
    for nz, ny, nx in product(range(s_diag[2]), range(s_diag[1]),range(s_diag[0])):
        icell = nx + ny * s_diag[0] + nz * s_diag[0] * s_diag[1]
        for iat in range(pw.nat):
            atoms_cart_new[:, icell * pw.nat + iat] = ( atoms_cart[:, iat]
                       + nx * cell[:, 0]  + ny * cell[:, 1] + nz * cell[:, 2] )

    return atoms_name_new, atoms_cart_new

def lat_to_reclat(cell):
    rec = np.zeros((3, 3))
    vol = np.dot(cell[:, 0], np.cross(cell[:, 1], cell[:, 2]))
    fac = 2 * np.pi / vol
    rec[:, 0] = np.cross(cell[:, 1], cell[:, 2]) * fac
    rec[:, 1] = np.cross(cell[:, 2], cell[:, 0]) * fac
    rec[:, 2] = np.cross(cell[:, 0], cell[:, 1]) * fac
    return rec

def make_fd_supercell(pw: PWInput, q_cry, ucart):
    import copy
    from math import floor, ceil
    from itertools import product
    """
    Generate nondiagonal supercell with finite displacements
    qe_input: a Pwscf() instance of an input file
    q_cry: perturbation crystal momentum, (m1/n1, m2/n2, m3/n3).
    ucart: displacement of atoms in the unit cell in cartesian alat units
    """
    # find fractional representation of q
    tol = 1E-10
    mlist = []
    nlist = []
    for i in range(3):
        found = False
        for n in range(1, 101):
            m = round(abs(q_cry[i]) * n)
            if abs(abs(q_cry[i]) - m / n) < tol:
                mlist.append(m)
                nlist.append(n)
                found = True
                break
        if not found:
            print(f"Unable to find fractional representation of q = {q[i]}")
            return None

    s_new = find_nondiagonal(mlist, nlist)
    multiplier = s_new[0, 0] * s_new[1, 1] * s_new[2, 2]
    cell_new = np.einsum('ij,xj->xi', s_new, pw.cell)
    nat_new = pw.nat * multiplier

    # Minkowski reduction of lattice vector: linear combination to make them
    # as short as possible
    cell_red = minkowski_reduce(cell_new)
    cell_cell = np.einsum('xi,xj->ij', pw.cell, pw.cell)
    redcell_cell = np.einsum('xi,xj->ij', cell_red, pw.cell)
    s_red = redcell_cell @ np.linalg.inv(cell_cell)
    s_red = np.rint(s_red)

    atoms_name_new, atoms_cart_new = atom_supercell(pw, s_new)

    # move atoms to [0, 1)^3 unit cell of cell_red
    atoms_cry_new = np.linalg.inv(cell_red) @ atoms_cart_new
    for iat in range(nat_new):
        for idir in range(3):
            atoms_cart_new[:, iat] -= floor(atoms_cry_new[idir, iat]) * cell_red[:, idir]

    # apply displacements
    s_diag = np.diag(s_new)
    for nz, ny, nx in product(range(s_diag[2]), range(s_diag[1]),range(s_diag[0])):
        icell = nx + ny * s_diag[0] + nz * s_diag[0] * s_diag[1]
        phase = np.exp(2j * np.pi * sum([x * y for x, y in zip(q_cry, (nx, ny, nz))]))
        for iat in range(pw.nat):
            atoms_cart_new[:, icell * pw.nat + iat] += (ucart[:, iat] * phase).real

    pw_super = PWInput(nat=nat_new, cell=cell_red, atoms_name=atoms_name_new,
                       atoms_cart=atoms_cart_new,
                       input_cards=copy.deepcopy(pw.input_cards),
                       input_namelists=copy.deepcopy(pw.input_namelists))

    if 'ATOMIC_POSITIONS' in pw_super.input_cards.keys():
        input_atmpos = ['ATOMIC_POSITIONS alat \n']
        for iat in range(pw_super.nat):
            line = (
                f" {atoms_name_new[iat]:4s} {atoms_cart_new[0, iat]:18.12f}"
                f" {atoms_cart_new[1, iat]:18.12f} {atoms_cart_new[2, iat]:18.12f}\n"
            )
            input_atmpos.append(line)
        pw_super.input_cards['ATOMIC_POSITIONS'] = input_atmpos

    if 'CELL_PARAMETERS' in pw_super.input_cards.keys():
        input_cell = ['CELL_PARAMETERS alat\n']
        for i in range(3):
            line = (
                f" {pw_super.cell[0,i]:18.12f} {pw_super.cell[1,i]:18.12f}"
                f" {pw_super.cell[2,i]:18.12f}\n"
            )
            input_cell.append(line)
        pw_super.input_cards['CELL_PARAMETERS'] = input_cell

    if 'K_POINTS' in pw_super.input_cards.keys():
        nks = np.array([int(x) for x in pw.input_cards['K_POINTS'][1].split()[:3]])
        dks = np.linalg.norm(lat_to_reclat(pw.cell), axis=0) / nks

        nks_new = np.linalg.norm(lat_to_reclat(pw_super.cell), axis=0) / min(dks)
        nks_new = [ceil(x) for x in nks_new]

        input_kpt = ['K_POINTS automatic\n']
        input_kpt.append(f" {nks_new[0]:3d}{nks_new[1]:3d}{nks_new[2]:3d}  0 0 0")
        pw_super.input_cards['K_POINTS'] = input_kpt

    # update namelists
    try:
        pw_super.input_namelists['SYSTEM']['ibrav'] = "0"
        pw_super.input_namelists['SYSTEM']['nat'] = f"{pw_super.nat}"
    except KeyError as e:
        print("KeyError in pw_super.input_namelists, keyword {e.args[0]}")

    return pw_super
