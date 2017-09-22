import numpy as np
#do not extrapolate
def values_from_table(mass, mass_table, value_table, consts):
    """
    """
    value = np.zeros(mass.size)
    for (idx, imass) in enumerate(mass):
        # Check if mass is larger than largest value in table and set to constant
        if imass > mass_table.max():
            value[idx] = 10**-6
            continue

        # check if any of the masses are in the table exactly
        if imass in mass_table:
            value[idx] = value_table[idx]
            continue

        # finally extrapolate if mass is in between to mass_table values
        for (idx_table, tmp) in enumerate(mass_table):
            if mass_table.size != idx_table + 1:
                if (mass_table[idx_table] < imass and mass_table[idx_table + 1] > imass):

                    value[idx] = value_table[idx_table] + consts[idx_table,0] * (imass - mass_table[idx_table]) + consts[idx_table,1] * (imass - mass_table[idx_table])**2 + consts[idx_table,2] *(imass - mass_table[idx_table])**3

                    continue
    return value
