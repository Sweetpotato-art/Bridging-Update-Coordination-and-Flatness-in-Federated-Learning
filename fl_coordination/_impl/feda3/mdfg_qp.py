"""Numerical recovery of the author's unchanged simplex minimum-norm QP.

The author builds V @ V.T in float32 before converting it to float64. With
millions of coordinates and a dependent fairness row this can turn a PSD Gram
matrix indefinite. Successful author calls remain untouched; recovery recomputes
the SAME inner products in float64, without ridge terms or eigenvalue clipping.
"""
import numpy as np


def gram64(vectors, block_size=262144):
    if vectors.ndim != 2 or not vectors.shape[0] or not vectors.shape[1]:
        raise ValueError('Expected a nonempty row matrix of FedMDFG directions')
    gram = np.zeros((len(vectors),len(vectors)),dtype=np.float64)
    for start in range(0,vectors.shape[1],block_size):
        block = np.asarray(vectors[:,start:start+block_size],dtype=np.float64)
        if not np.isfinite(block).all():
            raise FloatingPointError('Nonfinite FedMDFG input; precision recovery cannot repair data')
        gram += block @ block.T
    return (gram+gram.T)*.5


def certificate(gram,weights,tolerance=1e-8):
    weights = np.asarray(weights,dtype=np.float64)
    if weights.shape != (len(gram),) or not np.isfinite(weights).all():
        raise FloatingPointError('Invalid FedMDFG QP coefficients')
    feasibility = max(abs(float(weights.sum())-1.),max(0.,-float(weights.min())))
    scale = float(np.max(np.abs(gram)))
    normalized = gram/scale if scale>0 else gram
    gradient = normalized @ weights
    # A convex differentiable objective is minimized over the simplex iff its
    # first-order gap to the best vertex is zero. This also covers singular QPs.
    gap = float(weights @ gradient-gradient.min())
    if feasibility>tolerance or gap>tolerance or gap < -tolerance:
        raise FloatingPointError('FedMDFG precision recovery failed feasibility/optimality certificate')
    return dict(constraint_error=feasibility,normalized_optimality_gap=gap,
                objective=float(.5*weights @ gram @ weights))


def solve_gram(gram):
    from cvxopt import matrix, solvers
    gram = np.asarray(gram,dtype=np.float64)
    scale = float(np.max(np.abs(gram)))
    if not np.isfinite(gram).all() or not np.allclose(gram,gram.T,rtol=0,atol=1e-14*max(scale,1e-300)):
        raise FloatingPointError('Invalid Gram matrix in FedMDFG recovery')
    eigenvalues = np.linalg.eigvalsh(gram)
    if eigenvalues[0] < -1e-12*max(scale,1e-300):
        raise FloatingPointError('Float64 FedMDFG Gram matrix is unexpectedly indefinite')
    # Positive scalar objective scaling leaves all minimizers unchanged.
    normalized = gram/scale if scale>0 else gram
    n = len(gram)
    solution = solvers.qp(matrix(normalized),matrix(np.zeros(n)),matrix(-np.eye(n)),
                          matrix(np.zeros(n)),matrix(np.ones((1,n))),matrix(np.ones(1)),
                          options={'show_progress':False,'abstol':1e-11,'reltol':1e-11,
                                   'feastol':1e-11,'maxiters':200,'refinement':3})
    weights = np.asarray(solution['x']).reshape(-1)
    checked = certificate(gram,weights)
    if solution['status'] != 'optimal':
        raise FloatingPointError('Float64 FedMDFG QP did not converge: '+solution['status'])
    return weights,dict(solver='cvxopt_float64_recovery',iterations=solution['iterations'],
                        gram_scale=scale,min_eigenvalue_float64=float(eigenvalues[0]),**checked)


def solve(native,vectors):
    reason = None
    try:
        coefficients,optimal = native(vectors)
        if optimal and np.isfinite(coefficients).all():
            return coefficients,dict(solver='author_float32',recovered=False)
        reason = 'author_nonoptimal' if not optimal else 'author_nonfinite_coefficients'
    except ValueError as error:
        if not any(token in str(error) for token in ('domain error','Rank(A)','Rank([P')):
            raise
        reason = type(error).__name__+': '+str(error)
    except ArithmeticError as error:
        reason = type(error).__name__+': '+str(error)
    weights,diagnostics = solve_gram(gram64(vectors))
    return weights,dict(diagnostics,recovered=True,trigger=reason)
