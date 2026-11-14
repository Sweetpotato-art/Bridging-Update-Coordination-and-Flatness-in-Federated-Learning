"""Load pinned numerical functions without author CLIs or unrelated training imports."""
import ast
import copy
import importlib.util
from types import SimpleNamespace

import torch
from torch import nn, optim
from torch.autograd import Variable
from fl_coordination.dependencies import author_file


def functions(path, names, namespace):
    parsed = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise RuntimeError('Pinned author function names changed')
    code = ast.Module(body=nodes, type_ignores=[])
    exec(compile(code, str(path), 'exec'), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def awa_functions():
    namespace = dict(torch=torch, nn=nn, optim=optim, copy=copy, Variable=Variable)
    result = functions(author_file('FedAWA/server_funct.py'), ['_cost_matrix','to_var','fedawa'], namespace)
    # The author helper unconditionally uses the default CUDA device when one
    # exists. Keep the tensor on its requested device; numerical math is unchanged.
    namespace['to_var'] = lambda x, requires_grad=True: Variable(x, requires_grad=requires_grad)
    return result


def mdfg_functions():
    path = author_file('FedMDFG/fedplat/algorithm/common/utils.py')
    spec = importlib.util.spec_from_file_location('_feda3_author_mdfg_utils', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
