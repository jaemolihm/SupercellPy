import numpy as np


class PWInput:
    def __init__(self, nat, cell, atoms_name, atoms_cart, input_namelists={}, input_cards={}):
        if len(atoms_name) != nat:
            raise ValueError
        if atoms_cart.shape != (3, nat):
            raise ValueError

        # a_1 = cell[:,0], a_2 = cell[:,1], a_3 = cell[:,2]
        self.nat = nat
        self.cell = cell
        self.atoms_name = atoms_name
        self.atoms_cart = atoms_cart
        self.input_namelists = input_namelists
        self.input_cards = input_cards

    @classmethod
    def from_file(cls, filename):
        input_namelists = {}
        input_cards = {}
        variables_set = False
        with open(filename, 'r') as f:
            line = f.readline()
            while line:
                if line.strip() == '':
                    pass
                elif line.strip()[0] == '&':
                    # Namelist
                    namelist = line.strip()[1:].upper()
                    input_namelists[namelist] = {}
                    while True:
                        line = f.readline()
                        if line.strip() == '/':
                            break
                        keyword, value = line.strip().split('=')
                        keyword = keyword.strip()
                        value = value.strip()
                        if keyword[0] in ['!', '#']:
                            continue
                        input_namelists[namelist][keyword] = value
                else:
                    # Done reading namelists
                    if not variables_set:
                        variables_set = True
                        try:
                            ntyp = int(input_namelists['SYSTEM']['ntyp'])
                            nat = int(input_namelists['SYSTEM']['nat'])
                            ibrav = int(input_namelists['SYSTEM']['ibrav'])
                        except KeyError as e:
                            print(f"Error: keyword {e.args[0]} not found in file")
                            return None
                    if ibrav != 0:
                        print("Error: ibrav must be 0")
                        return None
                    # Read cards
                    card = line.strip().split()[0].strip().upper()
                    if card == 'CELL_PARAMETERS':
                        celltype = line.strip().split()[1].strip().lower()
                        input_cards[card] = [line] + [f.readline() for _ in range(3)]
                    elif card == 'ATOMIC_SPECIES':
                        input_cards[card] = [line] + [f.readline() for _ in range(ntyp)]
                    elif card == 'ATOMIC_POSITIONS':
                        atmtype = line.strip().split()[1].strip().lower()
                        input_cards[card] = [line] + [f.readline() for _ in range(nat)]
                    elif card == 'K_POINTS':
                        ktype = line.strip().split()[1].strip().lower()
                        if ktype == 'automatic':
                            input_cards[card] = [line] + [f.readline()]
                        else:
                            raise NotImplementedError("K_POINTS : Only AUTOMATIC is implemented.")

                line = f.readline()

        # parse cell parameters
        cell = np.zeros((3, 3))
        if celltype == 'alat':
            for i in range(3):
                line = input_cards['CELL_PARAMETERS'][i + 1]
                cell[:, i] = [float(x) for x in line.split()]
        else:
            raise NotImplementedError("CELL_PARAMETERS: only celltype alat is implemented")

        # parse atomic positions
        atoms_name = []
        atoms_cart = np.zeros((3, nat))
        if atmtype == 'alat':
            for iatm in range(nat):
                line = input_cards['ATOMIC_POSITIONS'][iatm + 1]
                data = line.split()
                atoms_name.append(data[0].strip())
                atoms_cart[:, iatm] = [float(x) for x in data[1:4]]
        elif atmtype == 'crystal':
            for iatm in range(nat):
                line = input_cards['ATOMIC_POSITIONS'][iatm + 1]
                data = line.split()
                atoms_name.append(data[0].strip())
                atoms_cart[:, iatm] = [float(x) for x in data[1:4]]
            # crystal to alat
            atoms_cart = cell @ atoms_cart
        else:
            raise NotImplementedError("ATOMIC_POSITIONS : Only alat and crystal is implemented.")

        return cls(nat=nat, cell=cell, atoms_name=atoms_name, atoms_cart=atoms_cart,
                   input_namelists=input_namelists, input_cards=input_cards)

    def write_file(self, filename):
        f = open(filename, 'w')
        # namelist_list = ['CONTROL', 'SYSTEM', 'ELECTRONS', 'IONS', 'CELL']
        for namelist in self.input_namelists.keys():
            f.write(f" &{namelist}\n")
            for keyword in self.input_namelists[namelist].keys():
                value = self.input_namelists[namelist][keyword]
                f.write(f"    {keyword:12s} = {value}\n")
            f.write(" /\n")
        for card in self.input_cards.keys():
            for line in self.input_cards[card]:
                f.write(line)
        f.close()

    def get_alat(self):
        try:
            alat = self.input_namelists['SYSTEM']['celldm(1)']
            return float(alat.strip().strip(','))
        except KeyError:
            print("celldm(1) not found. Return alat = 1.0.")
            return 1.0
