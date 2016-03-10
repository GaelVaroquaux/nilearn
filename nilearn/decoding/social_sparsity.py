"""
(F)ISTA solver for social sparsity: approximate overlapping group lasso

"""
# Author: VAROQUAUX Gael,
# License: simplified BSD

from math import sqrt

import numpy as np

from .fista import _check_lipschitz_continuous
from .objective_functions import (_unmask, _logistic_loss_grad,
    _logistic_loss_lipschitz_constant, _squared_loss_grad,
    spectral_norm_squared)


def fista(f1_grad, f2_prox, lipschitz_constant, w_size,
           init=None, max_iter=1000, tol=1e-4,
           check_lipschitz=False, callback=None,
           verbose=2, fista=True):
    """Generic FISTA solver.

    Minimizes the a sum `f + g` of two convex functions f (smooth)
    and g (proximable nonsmooth).

    Parameters
    ----------
    f1_grad : callable(w) -> np.array
        Gradient of smooth part of energy

    f2_prox : callable(w, stepsize) -> w
        Proximal-like operator of non-smooth part of energy (f2).

    lipschitz_constant : float
        Lipschitz constant of gradient of f1_grad.

    w_size : int
        Size of the solution. f1, f2, f1_grad, f2_prox (fixed l, tol) must
        accept a w such that w.shape = (w_size,).

    tol : float
        Tolerance on the (primal) cost function.

    init : dict-like, optional (default None)
        Dictionary of initialization parameters. Possible keys are 'w',
        'stepsize', 'z', 't', etc.

    callback : callable(dict) -> bool
        Function called on every iteration. If it returns True, then the loop
        breaks.

    max_iter : int
        Maximum number of iterations for the solver.

    fista : boolean, default: True
        If false, do an ista
    Returns
    -------
    w : ndarray, shape (w_size,)
       A minimizer for `f + g`.

    solver_info : float
        Solver information, for warm starting.

    history : array of floats
        dw computed on every iteration.

    Notes
    -----
    This ISTA implementation uses the weights as stopping criteria
    """

    # initialization
    if init is None:
        init = dict()
    w = init.get('w', np.zeros(w_size))
    z = init.get("z", w.copy())
    t = init.get("t", 1.)
    stepsize = init.get("stepsize", 1. / lipschitz_constant)

    # check Lipschitz continuity of gradient of smooth part
    if check_lipschitz:
        _check_lipschitz_continuous(f1_grad, w_size, lipschitz_constant)

    # aux variables
    ista_step = not fista
    stepsize = 1. / lipschitz_constant
    history = []
    w_old = np.empty_like(w)

    # FISTA loop
    for i in range(max_iter):

        # forward (gradient) step
        gradient_buffer = f1_grad(z)

        # backward (prox) step
        w = f2_prox(z - stepsize * gradient_buffer, stepsize)

        # Now check our stopping criteria
        if not (i % 5):
            # Every 5 iterations, recompute a scale
            scale = np.abs(w).max()
            # Regularize the scale if it gets very small
            scale = max(scale, 1e-7)
        dw = np.abs(w - w_old).max() / scale
        history.append(dw)

        # invoke callback
        if verbose:
            print('FISTA: Iteration % 2i/%2i: dw % 4.4e' % (
                  i + 1, max_iter, dw))
        if callback and callback(locals()):
            break
        if i > 5 and max(history[-5:]) < tol:
            if verbose:
                print("\tConverged (|dw| < %g)" % tol)
            break

        # ISTA / FISTA dance
        if ista_step:
            z = w
        else:
            t0 = t
            t = 0.5 * (1. + sqrt(1. + 4. * t * t))
            z = w + ((t0 - 1.) / t) * (w - w_old)
        w_old[:] = w


    init = dict(w=w.copy(), z=z, t=t, stepsize=stepsize)
    return w, history, init


def _neighboorhood_norm(img, side_weights=.7):
    " Return the squared norm averaged on 3D 6-neighboorhood"
    # Our stride tricks only work on C-contiguous arrays

    # Compute the norm on the groups
    grp_norms = img ** 2
    weighted_grps = side_weights * grp_norms
    # Sum along the first axis:
    grp_norms[:-1] += weighted_grps[1:]
    grp_norms[1:] += weighted_grps[:-1]

    # Sum along the second axis:
    grp_norms[:, :-1] += weighted_grps[:, 1:]
    grp_norms[:, 1:] += weighted_grps[:, :-1]

    # Sum along the third axis:
    grp_norms[:, :, :-1] += weighted_grps[:, :, 1:]
    grp_norms[:, :, 1:] += weighted_grps[:, :, :-1]

    # Normalization factor, so that everything sums to 1
    # For 6 neighboors
    factor = (1 + 6 * side_weights)
    grp_norms[1:-1, 1:-1, 1:-1] /= factor

    # Now the edges (only 3% of the time is spent below for large arrays)
    # For 5 neighboors (6 faces of a cube)
    factor = (1 + 5 * side_weights)
    grp_norms[   0, 1:-1, 1:-1] /= factor
    grp_norms[  -1, 1:-1, 1:-1] /= factor
    grp_norms[1:-1,    0, 1:-1] /= factor
    grp_norms[1:-1,   -1, 1:-1] /= factor
    grp_norms[1:-1, 1:-1,    0] /= factor
    grp_norms[1:-1, 1:-1,   -1] /= factor
    # For 4 neighboors (12 edges of a cube)
    factor = (1 + 4 * side_weights)
    grp_norms[   0,    0, 1:-1] /= factor
    grp_norms[   0,   -1, 1:-1] /= factor
    grp_norms[  -1,    0, 1:-1] /= factor
    grp_norms[  -1,   -1, 1:-1] /= factor
    grp_norms[1:-1,    0,    0] /= factor
    grp_norms[1:-1,    0,   -1] /= factor
    grp_norms[1:-1,   -1,    0] /= factor
    grp_norms[1:-1,   -1,   -1] /= factor
    grp_norms[   0, 1:-1,    0] /= factor
    grp_norms[   0, 1:-1,   -1] /= factor
    grp_norms[  -1, 1:-1,    0] /= factor
    grp_norms[  -1, 1:-1,   -1] /= factor
    # For 3 neighboors (8 vertices of a cube)
    factor = (1 + 3 * side_weights)
    grp_norms[ 0,  0,  0] /= factor
    grp_norms[ 0, -1,  0] /= factor
    grp_norms[-1,  0,  0] /= factor
    grp_norms[-1, -1,  0] /= factor
    grp_norms[ 0,  0, -1] /= factor
    grp_norms[ 0, -1, -1] /= factor
    grp_norms[-1,  0, -1] /= factor
    grp_norms[-1, -1, -1] /= factor

    return grp_norms


def _prox_social_sparsity(img, alpha):
    """Social sparsity on 3x3x3 groups, as defined by eq 4 of Kowalski et
    al, 'Social Sparsity...'"""
    grp_norms = _neighboorhood_norm(img)
    # To avoid side effects, assign the raw img values on the side
    grp_norms = np.sqrt(grp_norms, out=grp_norms)
    shrink = np.zeros(img.shape)
    img_nz = (img != 0)
    shrink[img_nz] = (1 - alpha / grp_norms[img_nz]).clip(0)

    return img * shrink


def social_solver(X, y, alpha, mask, loss=None, max_iter=100,
                lipschitz_constant=None, init=None,
                tol=1e-4, callback=None, verbose=1):
    """Solver for social-sparsity models.

    Can handle least squares (mean squared error --a.k.a mse) or logistic
    regression. The same solver works for both of these losses.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix.

    y : ndarray, shape (n_samples,)
        Target / response vector.

    alpha : float
        Constant that scales the overall regularization term. Defaults to 1.0.

    mask : ndarray, shape (nx, ny, nz)
        The support of this mask defines the ROIs being considered in
        the problem.

    max_iter : int
        Defines the iterations for the solver. Defaults to 100

    tol : float
        Defines the tolerance for convergence, in terms of relative
        max norm change on the coefficents. Defaults to 1e-3.

    loss : string
        Loss model for regression. Can be "mse" (for squared loss) or
        "logistic" (for logistic loss).

    lipschitz_constant : float, optional (default None)
        Lipschitz constant (i.e an upper bound of) of gradient of smooth part
        of the energy being minimized. If no value is specified (None),
        then it will be calculated.

    callback : callable(dict) -> bool, optional (default None)
        Function called at the end of every energy descendent iteration of the
        solver. If it returns True, the loop breaks.

    Returns
    -------
    w : ndarray, shape (n_features,)
       The solution vector (Where `w_size` is the size of the support of the
       mask.)

    objective : array of floats
        Objective function (fval) computed on every iteration.

    solver_info: float
        Solver information, for warm start.

    """
    # sanitize loss
    if loss not in ["mse", "logistic"]:
        raise ValueError("'%s' loss not implemented. Should be 'mse' or "
                         "'logistic" % loss)
    # in logistic regression, we fit the intercept explicitly
    w_size = X.shape[1] + int(loss == "logistic")

    # function to compute derivative of f1
    if loss == "logistic":
        def f1_grad(w):
            return _logistic_loss_grad(X, y, w)
    else:
        def f1_grad(w):
            return _squared_loss_grad(X, y, w)

    # Lipschitz constant of f1_grad
    if lipschitz_constant is None:
        if loss == "mse":
            lipschitz_constant = 1.05 * spectral_norm_squared(X)
        else:
            lipschitz_constant = 1.1 * _logistic_loss_lipschitz_constant(X)

    # proximal operator of nonsmooth proximable part of energy (f2)
    if loss == "mse":
        def f2_prox(w, stepsize):
            out = _prox_social_sparsity(_unmask(w, mask),
                                        alpha * stepsize)
            return out[mask]
    else:
        # Deal with intercept
        def f2_prox(w, stepsize):
            out = _prox_social_sparsity(_unmask(w[:-1], mask),
                                        alpha * stepsize)
            return np.append(out[mask], w[-1])

    # invoke FISTA solver
    w, obj, init = fista(
        f1_grad, f2_prox, lipschitz_constant, w_size,
        tol=tol, init=init, verbose=verbose,
        max_iter=max_iter, callback=callback)

    return w, obj, init


def social_solver_with_l1(X, y, alpha, l1_ratio, mask,
                loss=None, max_iter=100,
                lipschitz_constant=None, init=None,
                tol=1e-4, callback=None, verbose=1):
    # Hack to plug social-sparsity in SpaceNet
    return social_solver(X, y, alpha, mask,
                loss=loss, max_iter=max_iter,
                lipschitz_constant=lipschitz_constant, init=init,
                tol=tol, callback=callback, verbose=verbose)
