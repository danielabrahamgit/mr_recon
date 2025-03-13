"""
Everything Spherical Harmonics up to 5th order.
Author: Zachary Shah
"""

MAX_ORDER = 5
MAX_SH_COEFFICIENT_INDEX = 35

"""
LIST OF SPHERICAL BASES FUNCTIONS for r = [x, y, z]
"""
SH_BASES_FUNCTIONS = [
    # 0th order : 1 base
    lambda r: (r[0] * 0.0 + 1.0),

    # 1st order : 3 bases
    lambda r: r[0],
    lambda r: r[1],
    lambda r: r[2],

    # 2nd order : 5 bases
    lambda r: r[0] * r[1],
    lambda r: r[2] * r[1],
    lambda r: 3 * r[2]**2 - (r[0]**2 + r[1]**2 + r[2]**2),
    lambda r: r[0] * r[2],
    lambda r: r[0]**2 - r[1]**2,

    # 3rd order : 7 bases
    lambda r: 3 * r[1] * r[0]**2 - r[1]**3,
    lambda r: r[0] * r[1] * r[2],
    lambda r: (5 * r[2]**2 - (r[0]**2 + r[1]**2 + r[2]**2)) * r[1],
    lambda r: 5 * r[2]**3 - 3 * r[2] * (r[0]**2 + r[1]**2 + r[2]**2),
    lambda r: (5 * r[2]**2 - (r[0]**2 + r[1]**2 + r[2]**2)) * r[0],
    lambda r: r[2] * r[0]**2 - r[2] * r[1]**2,
    lambda r: r[0]**3 - 3 * r[0] * r[1]**2,

    # 4th order : 9 bases
    lambda r: r[0] * r[1] * (r[0]**2 - r[1]**2),
    lambda r: r[1] * r[2] * (3 * r[0]**2 - r[1]**2),
    lambda r: r[0] * r[1] * (7 * r[2]**2 - (r[0]**2 + r[1]**2 + r[2]**2)),
    lambda r: r[1] * (7 * r[2]**3 - 3 * r[2] * (r[0]**2 + r[1]**2 + r[2]**2)),
    lambda r: 35 * r[2]**4 - 30 * r[2]**2 * (r[0]**2 + r[1]**2 + r[2]**2) + 3 * ((r[0]**2 + r[1]**2 + r[2]**2)**2),
    lambda r: r[0] * (7 * r[2]**3 - 3 * r[2] * (r[0]**2 + r[1]**2 + r[2]**2)),
    lambda r: (r[0]**2 - r[1]**2) * (7 * r[2]**2 - (r[0]**2 + r[1]**2 + r[2]**2)),
    lambda r: r[0] * (r[0]**2 - 3 * r[1]**2) * r[2],
    lambda r: r[0]**2 * (r[0]**2 - 3 * r[1]**2) - r[1]**2 * (3 * r[0]**2 - r[1]**2),

    # 5th order : 11 bases
    lambda r: r[1] * (5 * r[0]**4 - 10 * r[0]**2 * r[1]**2 + r[1]**4),
    lambda r: r[0] * r[1] * r[2] * (r[0]**2 - r[1]**2),
    lambda r: r[1] * ((r[0]**2 + r[1]**2 + r[2]**2) - 9 * r[2]**2) * (r[1]**2 - 3 * r[0]**2),
    lambda r: r[0] * r[1] * r[2] * (3 * r[2]**2 - (r[0]**2 + r[1]**2 + r[2]**2)),
    lambda r: r[1] * ((r[0]**2 + r[1]**2 + r[2]**2)**2 - 14 * (r[0]**2 + r[1]**2 + r[2]**2) * r[2]**2 + 21 * r[2]**4),
    lambda r: 63 * r[2]**5 - 70 * r[2]**3 * (r[0]**2 + r[1]**2 + r[2]**2) + 15 * r[2] * (r[0]**2 + r[1]**2 + r[2]**2)**2,
    lambda r: r[0] * ((r[0]**2 + r[1]**2 + r[2]**2)**2 - 14 * (r[0]**2 + r[1]**2 + r[2]**2) * r[2]**2 + 21 * r[2]**4),
    lambda r: r[2] * (3 * r[2]**2 - (r[0]**2 + r[1]**2 + r[2]**2)) * (r[0]**2 - r[1]**2),
    lambda r: r[0] * ((r[0]**2 + r[1]**2 + r[2]**2) - 9 * r[2]**2) * (3 * r[1]**2 - r[0]**2),
    lambda r: r[2] * (r[0]**4 - 6 * r[0]**2 * r[1]**2 + r[1]**4),
    lambda r: r[0]**5 - 10 * r[0]**3 * r[1]**2 + 5 * r[0] * r[1]**4
]


# Names of coefficients
SH_COEFFICIENT_NAMES = [
    # 0th
    "1",

    # 1st
    "x", 
    "y", 
    "z",

    # 2nd
    "xy", 
    "zy", 
    "3z^2-(x^2+y^2+z^2)", 
    "xz", 
    "x^2-y^2",

    # 3rd
    "3x^2y-y^3", 
    "xyz", 
    "(5z^2-x^2+y^2+z^2)y", 
    "5z^3-3z(x^2+y^2+z^2)",
    "(5z^2-(x^2+y^2+z^2))x", 
    "zx^2 - zy^2", 
    "x^3-3xy^2",

    # 4th
    "xy(x^2-y^2)",
    "yz(3x^2-y^2)",
    "xy(7z^2-(x^2+y^2+z^2))",
    "y(7z^3-3z(x^2+y^2+z^2))",
    "35z^4-30z^2(x^2+y^2+z^2)+3((x^2+y^2+z^2)^2)",
    "x(7z^3-3z(x^2+y^2+z^2))",
    "(x^2-y^2)(7z^2-(x^2+y^2+z^2))",
    "x(x^2-3y^2)z",
    "x^2(x^2-3y^2)-y^2(3x^2-y^2)",

    # 5th
    "y(5x^4-10x^2y^2+y^4)",
    "xyz(x^2-y^2)",
    "y((x^2+y^2+z^2)-9z^2)(y^2-3x^2)",
    "xyz(3z^2-(x^2+y^2+z^2))",
    "y((x^2+y^2+z^2)^2-14(x^2+y^2+z^2)z^2+21z^4)",
    "63z^5-70z^3(x^2+y^2+z^2)+15z(x^2+y^2+z^2)^2",
    "x((x^2+y^2+z^2)^2-14(x^2+y^2+z^2)z^2+21z^4)",
    "z(3z^2-(x^2+y^2+z^2))(x^2-y^2)",
    "x((x^2+y^2+z^2)-9z^2)(3y^2-x^2)",
    "z(x^4-6x^2y^2+y^4)",
    "x^5-10x^3y^2+5xy^4",
]

# Units for each coefficient for skope
SH_COEFFICIENT_UNITS = ["rad"] + 3 * ["rad/m"] + 5 * ["rad/m^2"] + 7 * ["rad/m^3"] + 9 * ["rad/m^4"] + 11 * ["rad/m^5"]

# Coefficient Names for XY (Assume Z=0 and insert "0" as name for placeholder where basis function is 0 forall XY)
SH_2D_COEFFICIENT_NAMES = [
    # 0th
    "1",

    # 1st
    "x", 
    "y", 
    "0",

    # 2nd
    "xy", 
    "0", 
    "-(x^2+y^2)", 
    "0", 
    "x^2-y^2",

    # 3rd
    "3x^2y-y^3", 
    "0", 
    "(-x^2+y^2)y", 
    "0",
    "(x^2+y^2)x", 
    "0", 
    "x^3-3xy^2",

    # 4th
    "xy(x^2-y^2)",
    "0",
    "-xy(x^2+y^2)",
    "0",
    "3(x^2+y^2)^2",
    "0",
    "-(x^4-y^4)",
    "0",
    "x^4-y^2(3x^2-y^2)",

    # 5th
    "y(5x^4-10x^2y^2+y^4)",
    "0",
    "y(x^2+y^2)(y^2-3x^2)",
    "0",
    "y(x^2+y^2)^2",
    "0",
    "x((x^2+y^2)^2",
    "0",
    "x(x^2+y^2))(3y^2-x^2)",
    "0",
    "x^5-10x^3y^2+5xy^4",
]

# Indicies where 2D coefficients would be non-zero
SH_2D_COEFFICIENT_INDICES = [
    0, # 0th - 1
    1, 2,  # 1st - 3
    4, 6, 8, # 2nd - 6
    9, 11, 13, 15, # 3rd - 10
    16, 18, 20, 22, 24, # 4th - 15
    25, 27, 29, 31, 33, 35, # 5th - 21
]