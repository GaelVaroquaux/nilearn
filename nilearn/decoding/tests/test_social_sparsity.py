"""
Test the social sparsity implementation.
"""
import numpy as np

from nose.tools import assert_equal

from ..social_sparsity import _neighboorhood_norm, _prox_social_sparsity
from ..proximal_operators import _prox_l1

def test_neighboorhood_norm():
    # A trivial test: check that we are getting the size of the groups
    # ones, if we input only ones
    img1 = np.ones((7, 6, 5))
    grp_norms = _neighboorhood_norm(img1)
    np.testing.assert_array_equal(grp_norms,
                                  np.ones((5, 4, 3)))

    # Second somewhat trivial test: check that if all the elements of an
    # array, are different, all the local average are different (to check
    # that we are not summing in the same direction twice
    img2 = np.arange(7 * 6 * 5).reshape((7, 6, 5))
    grp_norms = _neighboorhood_norm(img2)
    assert_equal(np.unique(grp_norms).size, grp_norms.size)


def test_prox_social_sparsity():
    # Check that on a constant image, social_sparsity is equivalent to l1
    img1 = -2.3 * np.ones((7, 6, 5))
    for alpha in (.1, .9, 1, 2):
        np.testing.assert_array_equal(_prox_social_sparsity(img1, alpha),
                                      _prox_l1(img1, alpha))

