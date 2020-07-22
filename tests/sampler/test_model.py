import unittest
from gwemlightcurves.sampler import model


class TestModel(unittest.TestCase):
    def test_Me2017_model_ejecta(self):
        # This test only makes sure the model runs and that the results pass a
        # few basic consistency checks. The test could be extended to check that
        # the results also make sense physically.
        t_days, L_bol, mag = model.Me2017_model_ejecta(mej=0.01,
                                                       vej=0.1,
                                                       beta=3,
                                                       kappa_r=20)
        self.assertEqual(len(t_days), len(L_bol))
        self.assertEqual(len(t_days), mag.shape[1])
        self.assertEqual(
            len(mag), 9,
            ("The model should return magnitudes in 9 frequency bands, "
             "but got {}.".format(len(mag))))


if __name__ == '__main__':
    unittest.main()
