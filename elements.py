"""Module 4 - Defining the elements"""
"""Element data, periodic-table grid positions, and category classification."""

# (symbol, name, mass number A). A is the common/most stable isotope's integer
# mass; neutrons = A - Z. Used later by the Bohr renderer.
_DATA = """
H Hydrogen 1|He Helium 4|Li Lithium 7|Be Beryllium 9|B Boron 11|C Carbon 12|
N Nitrogen 14|O Oxygen 16|F Fluorine 19|Ne Neon 20|Na Sodium 23|Mg Magnesium 24|
Al Aluminium 27|Si Silicon 28|P Phosphorus 31|S Sulfur 32|Cl Chlorine 35|Ar Argon 40|
K Potassium 39|Ca Calcium 40|Sc Scandium 45|Ti Titanium 48|V Vanadium 51|
Cr Chromium 52|Mn Manganese 55|Fe Iron 56|Co Cobalt 59|Ni Nickel 59|Cu Copper 64|
Zn Zinc 65|Ga Gallium 70|Ge Germanium 73|As Arsenic 75|Se Selenium 79|Br Bromine 80|
Kr Krypton 84|Rb Rubidium 85|Sr Strontium 88|Y Yttrium 89|Zr Zirconium 91|
Nb Niobium 93|Mo Molybdenum 96|Tc Technetium 98|Ru Ruthenium 101|Rh Rhodium 103|
Pd Palladium 106|Ag Silver 108|Cd Cadmium 112|In Indium 115|Sn Tin 119|
Sb Antimony 122|Te Tellurium 128|I Iodine 127|Xe Xenon 131|Cs Cesium 133|
Ba Barium 137|La Lanthanum 139|Ce Cerium 140|Pr Praseodymium 141|Nd Neodymium 144|
Pm Promethium 145|Sm Samarium 150|Eu Europium 152|Gd Gadolinium 157|Tb Terbium 159|
Dy Dysprosium 163|Ho Holmium 165|Er Erbium 167|Tm Thulium 169|Yb Ytterbium 173|
Lu Lutetium 175|Hf Hafnium 178|Ta Tantalum 181|W Tungsten 184|Re Rhenium 186|
Os Osmium 190|Ir Iridium 192|Pt Platinum 195|Au Gold 197|Hg Mercury 201|
Tl Thallium 204|Pb Lead 207|Bi Bismuth 209|Po Polonium 209|At Astatine 210|
Rn Radon 222|Fr Francium 223|Ra Radium 226|Ac Actinium 227|Th Thorium 232|
Pa Protactinium 231|U Uranium 238|Np Neptunium 237|Pu Plutonium 244|Am Americium 243|
Cm Curium 247|Bk Berkelium 247|Cf Californium 251|Es Einsteinium 252|Fm Fermium 257|
Md Mendelevium 258|No Nobelium 259|Lr Lawrencium 266|Rf Rutherfordium 267|
Db Dubnium 268|Sg Seaborgium 269|Bh Bohrium 270|Hs Hassium 277|Mt Meitnerium 278|
Ds Darmstadtium 281|Rg Roentgenium 282|Cn Copernicium 285|Nh Nihonium 286|
Fl Flerovium 289|Mc Moscovium 290|Lv Livermorium 293|Ts Tennessine 294|Og Oganesson 294
"""

# ELEMENTS[Z] = (symbol, name, A); index 0 is a dummy so that index == Z.
ELEMENTS = [None]
for _item in _DATA.replace("\n", "").split("|"):
    _sym, _name, _a = _item.strip().split()
    ELEMENTS.append((_sym, _name, int(_a)))
assert len(ELEMENTS) == 119, "expected 118 elements"


def grid_pos(z):
    """(row, col) of element z in the 18-column table.
    Rows 0-6 are periods 1-7; lanthanides sit in row 8, actinides in row 9
    (row 7 is a visual gap)."""
    if z == 1:          return 0, 0
    if z == 2:          return 0, 17
    if 3 <= z <= 4:     return 1, z - 3
    if 5 <= z <= 10:    return 1, z + 7
    if 11 <= z <= 12:   return 2, z - 11
    if 13 <= z <= 18:   return 2, z - 1
    if 19 <= z <= 36:   return 3, z - 19
    if 37 <= z <= 54:   return 4, z - 37
    if 55 <= z <= 56:   return 5, z - 55
    if 57 <= z <= 71:   return 8, z - 55          # La..Lu -> cols 2..16
    if 72 <= z <= 86:   return 5, z - 69          # Hf..Rn -> cols 3..17
    if 87 <= z <= 88:   return 6, z - 87
    if 89 <= z <= 103:  return 9, z - 87          # Ac..Lr -> cols 2..16
    if 104 <= z <= 118: return 6, z - 101         # Rf..Og -> cols 3..17
    raise ValueError(z)


def category(z):
    """Category name used for colouring."""
    if z in (3, 11, 19, 37, 55, 87):            return "alkali"
    if z in (4, 12, 20, 38, 56, 88):            return "alkaline"
    if z in (2, 10, 18, 36, 54, 86, 118):       return "noble"
    if z in (9, 17, 35, 53, 85, 117):           return "halogen"
    if z in (5, 14, 32, 33, 51, 52):            return "metalloid"
    if z in (1, 6, 7, 8, 15, 16, 34):           return "nonmetal"
    if 57 <= z <= 71:                           return "lanthanide"
    if 89 <= z <= 103:                          return "actinide"
    if 21 <= z <= 30 or 39 <= z <= 48 or 72 <= z <= 80 or 104 <= z <= 112:
        return "transition"
    return "post"                               # remaining metals (Al, Ga, Pb, ...)
