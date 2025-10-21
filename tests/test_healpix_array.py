import healpy as hp
import numpy as np
import numpy.testing as npt
import pytest

from hpxutils.healpix_array import HealpixMask


class TestHealpixMap:
    """Tests for HealpixMap class."""

    sd_map = np.zeros(12, dtype=bool)
    sd_map[0] = 1

    hd_map = np.zeros(48, dtype=bool)
    hd_map[0] = 1
    hd_map[2] = 1
    hd_map[-1] = 1

    hd_map_bis = np.zeros(48, dtype=bool)
    hd_map_bis[0] = 1
    hd_map_bis[-2] = 1

    def test_initialization_wrongdtype(self):
        with pytest.raises(TypeError) as excinfo:
            HealpixMask(np.ones(12, dtype=int), nest=True)
        assert excinfo.type is TypeError

    def test_initialization_wrongnpix(self):
        with pytest.raises(TypeError) as excinfo:
            HealpixMask(np.ones(11, dtype=bool), nest=True)
        assert excinfo.type is TypeError

    def test_initialization_nest(self):
        mask = HealpixMask(self.sd_map, nest=True)
        assert mask.nest
        assert mask.nside == 1
        npt.assert_array_equal(actual=mask, desired=self.sd_map)

    def test_initialization_ring(self):
        mask = HealpixMask(self.sd_map, nest=False)
        assert mask.nest
        assert mask.nside == 1
        npt.assert_array_equal(actual=mask, desired=hp.reorder(self.sd_map, r2n=True))

    def test_logical_and_different_nside(self):
        hd_mask = HealpixMask(mask=self.hd_map, nest=True)
        sd_mask = HealpixMask(mask=self.sd_map, nest=True)
        expected_intersection = np.array([ True, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])  # fmt: skip
        # try both orders
        intermask = hd_mask & sd_mask
        npt.assert_array_equal(actual=intermask, desired=expected_intersection)
        intermask = sd_mask & hd_mask
        npt.assert_array_equal(actual=intermask, desired=expected_intersection)

    def test_logical_and_same_nside(self):
        hd_mask = HealpixMask(mask=self.hd_map, nest=True)
        hd_mask_bis = HealpixMask(mask=self.hd_map_bis, nest=True)
        expected_intersection = np.array([ True, False,  False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])  # fmt: skip
        # try both orders
        intermask = hd_mask & hd_mask_bis
        npt.assert_array_equal(actual=intermask, desired=expected_intersection)
        intermask = hd_mask_bis & hd_mask
        npt.assert_array_equal(actual=intermask, desired=expected_intersection)

    def test_logical_or_different_nside(self):
        hd_mask = HealpixMask(mask=self.hd_map, nest=True)
        sd_mask = HealpixMask(mask=self.sd_map, nest=True)
        expected_union = np.array([ True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True])  # fmt: skip
        # try both orders
        unionmask = hd_mask | sd_mask
        npt.assert_array_equal(actual=unionmask, desired=expected_union)
        unionmask = sd_mask | hd_mask
        npt.assert_array_equal(actual=unionmask, desired=expected_union)

    def test_logical_and_same_nside(self):
        hd_mask = HealpixMask(mask=self.hd_map, nest=True)
        hd_mask_bis = HealpixMask(mask=self.hd_map_bis, nest=True)
        expected_union = np.array([ True, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True])  # fmt: skip
        # try both orders
        unionmask = hd_mask | hd_mask_bis
        npt.assert_array_equal(actual=unionmask, desired=expected_union)
        unionmask = hd_mask_bis | hd_mask
        npt.assert_array_equal(actual=unionmask, desired=expected_union)
