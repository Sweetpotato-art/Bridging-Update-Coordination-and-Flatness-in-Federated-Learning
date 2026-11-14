"""Precision recovery for the unchanged centered MGDA quadratic program.

No diagonal shift, eigenvalue clipping, gradient rescaling or uniform-weight
fallback. Normal calls still execute the pinned author's solver unchanged.
"""
import numpy as np
import torch


def gram64(updates):
    n = len(updates)
    q = np.zeros((n, n), dtype=np.float64)
    for key in updates[0]:
        block = torch.stack([u[key].detach().reshape(-1).cpu().double() for u in updates])
        if not torch.isfinite(block).all():
            raise FloatingPointError('Nonfinite MGDA client direction')
        q += (block @ block.T).numpy()
    return (q + q.T) * .5


def centered_constraints(n, epsilon):
    # Exactly the inequalities in vendor/FedMGDAPlus/main/utils.py.
    eye = np.eye(n, dtype=float)
    c = np.concatenate((np.ones((n, 1)), eye, eye, -eye), axis=1)
    b = np.concatenate(([1.], np.zeros(n), np.full(n, 1./n-epsilon),
                        np.full(n, -1./n-epsilon)))
    return c, b


def check_solution(q, weights, epsilon=1., tolerance=1e-7):
    """Primal feasibility and a convex first-order optimality certificate.

    The linear minimization oracle solves the same box-constrained simplex,
    including the author's centered epsilon bound. Its gap is zero at a QP
    optimum, even for a singular Gram matrix with nonunique weights.
    """
    w = np.asarray(weights, dtype=float)
    n = len(q)
    lower, upper = max(0., 1./n-epsilon), min(1., 1./n+epsilon)
    feasibility = max(abs(w.sum()-1.), float(np.max(lower-w)), float(np.max(w-upper)), 0.)
    if w.shape != (n,) or not np.isfinite(w).all() or feasibility > tolerance:
        raise FloatingPointError('MGDA QP solution violates the original constraints')
    gradient = q @ w
    vertex = np.full(n, lower)
    remaining = 1. - n*lower
    for i in np.argsort(gradient, kind='stable'):
        amount = min(remaining, upper-lower)
        vertex[i] += amount
        remaining -= amount
    gap = float(gradient @ (w-vertex))
    if abs(remaining) > tolerance or gap > tolerance * max(1., np.max(abs(q))):
        raise FloatingPointError('MGDA QP solution failed its optimality certificate')
    return {'constraint_error':float(feasibility), 'optimality_gap':gap,
            'objective':float(.5*w @ q @ w)}


def solve_psd(q, epsilon=1.):
    """Solve the identical convex QP when strict positive definiteness fails."""
    from cvxopt import matrix, solvers
    c, b = centered_constraints(len(q), epsilon)
    result = solvers.qp(matrix(q), matrix(np.zeros(len(q))),
                        matrix(-c[:, 1:].T), matrix(-b[1:]),
                        matrix(c[:, :1].T), matrix(b[:1]),
                        options={'show_progress':False, 'abstol':1e-10,
                                 'reltol':1e-10, 'feastol':1e-10, 'maxiters':200})
    if result['status'] != 'optimal':
        raise FloatingPointError('MGDA PSD solver did not converge: '+result['status'])
    weights = np.asarray(result['x']).reshape(-1)
    certificate = check_solution(q, weights, epsilon)
    return weights, certificate


def recover(native, updates, diagnostics, epsilon=1.):
    q = gram64(updates)
    eigenvalues = np.linalg.eigvalsh(q)
    scale = max(1., float(np.linalg.norm(q, ord=2)))
    if eigenvalues[0] < -1e-12*scale:
        raise FloatingPointError('Float64 Gram matrix is unexpectedly indefinite')
    c, b = centered_constraints(len(updates), epsilon)
    try:
        weights = native.utils.quadprog.solve_qp(q, np.zeros(len(q)), c, b, meq=1)[0]
        certificate = check_solution(q, weights, epsilon)
        solver = 'quadprog_float64'
    except ValueError as error:
        if 'not positive definite' not in str(error):
            raise
        weights, certificate = solve_psd(q, epsilon)
        solver = 'cvxopt_psd_float64'
    if diagnostics is not None:
        diagnostics.update(solver=solver, min_eigenvalue_float64=float(eigenvalues[0]),
                           **certificate)
    return weights
