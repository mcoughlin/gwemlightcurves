import lalsimulation
import lal
from distutils.spawn import find_executable
from astropy.table import Table

class EOS4ParameterPiecewisePolytrope(object):
    """4-piece polytrope equation of state.
    """

    def __init__(self, EOS):
        """Initialize EOS and calculate a family of TOV stars.
        """
        #print(find_executable('polytrope_table.dat'))
        # load in polytop table
        polytable = Table.read(find_executable('polytrope_table.dat'), format='ascii')
        polytable = polytable[polytable['col1'] == EOS]
        lp_cgs = float(polytable['col2'])
        g1 = float(polytable['col3'])
        g2 = float(polytable['col4'])
        g3 = float(polytable['col5'])

        # lalsimulation uses SI units.
        lp_si = lp_cgs - 1.

        # Initialize with piecewise polytrope parameters (logp1 in SI units)
        eos = lalsimulation.SimNeutronStarEOS4ParameterPiecewisePolytrope(lp_si, g1, g2, g3)

        # This creates the interpolated functions R(M), k2(M), etc.
        # after doing many TOV integrations.
        self.fam = lalsimulation.CreateSimNeutronStarFamily(eos)

        # Get maximum mass for this EOS
        self.mmax = lalsimulation.SimNeutronStarMaximumMass(self.fam)/lal.MSUN_SI
    def radiusofm(self, m):
        """Radius in km.
        """
        try:
            r_SI = lalsimulation.SimNeutronStarRadius(m*lal.MSUN_SI, self.fam)
            return r_SI/1000.0
        except:
            return -1

    def k2ofm(self, m):
        """Dimensionless Love number.
        """
        try:
            return lalsimulation.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, self.fam)
        except:
            return -1

    def lambdaofm(self, m):
        """Dimensionless tidal deformability.
        """
        r = self.radiusofm(m)
        k2 = self.k2ofm(m)
        return (2./3.)*k2*( (lal.C_SI**2*r*1000.0)/(lal.G_SI*m*lal.MSUN_SI) )**5

    def maxmass(self):
        """Fuction to return the max mass
        """

        return self.mmax

