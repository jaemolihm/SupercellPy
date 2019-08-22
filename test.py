import numpy as np
from supercellpy import PWInput, make_fd_supercell
"""
Test example for using make_fd_supercell
One can also use sys.path.append("/group2/jmlim/program/SupercellPy")
"""

BOHR_TO_ANG = 0.529177

# read pw.x input to PWInput object
pw = PWInput.from_file('test.in')

# Define perturbation
# phonon crystal momentum
q = [0, 1/2, 1/3]
# phonon displacement mode
u = np.zeros((3, 2), dtype=complex)
u[:, 0] = [0.499528 + 0.024785j, 0.499240 + 0.024834j, -0.000288 + 0.000049j]
u[:, 1] = [0.488266 + 0.108347j, 0.487974 + 0.108347j, -0.000292 + 0.000000j]
# normalize phonon displacement
amplitude = 1E-3 # angstrom
alat = pw.get_alat()  # bohr
u *= amplitude / (alat * BOHR_TO_ANG) / np.linalg.norm(u)

# make supercell
pw_super = make_fd_supercell(pw, q, u)

# write pw.x input object for supercell
pw_super.write_file('test_super.in')

print(pw_super.cell)
print(pw_super.atoms_name)
for i in range(pw_super.nat):
    print(pw_super.atoms_cart[:, i])
