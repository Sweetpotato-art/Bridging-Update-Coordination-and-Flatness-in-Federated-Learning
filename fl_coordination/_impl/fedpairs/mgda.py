"""Centered MGDA quadratic program with precision recovery."""
import contextlib
import io


def coefficients(native, updates, *, robust=False, diagnostics=None):
    if diagnostics is not None:
        diagnostics['solver'] = 'author_quadprog'
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return native.utils.solve_centered_w(updates, epsilon=1.)
    except ValueError as error:
        if not robust or 'not positive definite' not in str(error):
            raise
        from .mgda_qp import recover
        return recover(native, updates, diagnostics, epsilon=1.)
