"""Numerical entry points for the pinned author coordinators."""
import ast
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import quadprog
import torch

from fl_coordination.dependencies import author_file
from fl_coordination._impl.feda3.upstream import functions


def fv(device):
    # Import only the server arithmetic and the model-dictionary operations it uses.
    path = author_file('FedFV/method/fedfv.py')
    math_path = author_file('FedFV/task/modelfuncs.py')
    tree = ast.parse(path.read_bytes())
    server = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Server')
    methods = [node for node in server.body if isinstance(node, ast.FunctionDef) and node.name in ('iterate', 'aggregate')]
    if len(methods) != 2:
        raise RuntimeError('Pinned FedFV server method is missing')
    required = {node.attr for node in ast.walk(ast.Module(body=methods,type_ignores=[]))
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == 'modelfuncs'}
    math_tree = ast.parse(math_path.read_bytes())
    definitions = {node.name: node for node in math_tree.body if isinstance(node, ast.FunctionDef)}
    requested = set(required)
    pending = list(required)
    while pending:
        name = pending.pop()
        for node in ast.walk(definitions[name]):
            if isinstance(node, ast.Name) and node.id in definitions and node.id not in requested:
                requested.add(node.id)
                pending.append(node.id)
    namespace = dict(torch=torch, np=np, device=device)
    exec(compile(ast.Module(body=[node for name,node in definitions.items()
                                 if name in requested], type_ignores=[]),str(math_path),'exec'),namespace)
    import copy
    import math
    scope = dict(torch=torch, copy=copy, math=math, modelfuncs=SimpleNamespace(**namespace))
    exec(compile(ast.Module(body=methods,type_ignores=[]),str(path),'exec'), scope)
    return SimpleNamespace(Server=type('AuthorFedFVServer',(),{node.name: scope[node.name] for node in methods}))


@lru_cache(maxsize=1)
def mgda():
    path = author_file('FedMGDAPlus/main/utils.py')
    return SimpleNamespace(utils=functions(path, ['solve_centered_w'],
        dict(torch=torch, np=np, quadprog=quadprog)))
