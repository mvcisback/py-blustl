import io
import pathlib
from math import ceil
from collections import defaultdict
from operator import itemgetter as ig

import stl
import yaml
import numpy as np
import funcy as fn
import sympy as sym
from lenses import lens

from magnum.game import Game, Meta, Specs, Model, Vars, set_time
from magnum.utils import dynamics_lipschitz

def parse_model(g):
    # Parse Model
    dt = float(g['model']['dt'])
    stl_var_map = fn.merge(
        {
        'input': [],
        'state': [],
        'env': []
    }, g['model']['vars'])
    stl_var_map['input'] = list(map(sym.Symbol, stl_var_map['input']))
    stl_var_map['state'] = list(map(sym.Symbol, stl_var_map['state']))
    stl_var_map['env'] = list(map(sym.Symbol, stl_var_map['env']))

    bounds = {k: v.split(",") for k, v in g["model"]["bounds"].items()}
    bounds = {
        sym.Symbol(k): (float(v[0][1:]), float(v[1][:-1]))
        for k, v in bounds.items()
    }
    steps = int(ceil(int(g['model']['time_horizon']) / dt))
    return Model(dt=dt, N=steps, vars=Vars(**stl_var_map), bounds=bounds, t=0)


def matrix_to_dyn_stl(A, B, C, model):
    """TODO: cleanup"""
    def to_terms(row, syms, t=stl.t_sym):
        return [stl.Var(c, s, t) for s, c in zip(syms, row) if c != 0]

    def row_to_stl(i, row):
        a_row, b_row, c_row = row
        terms = to_terms(a_row, model.vars.state)
        terms += to_terms(b_row, model.vars.input)
        terms += to_terms(c_row, model.vars.env)
        terms.append(stl.Var(-1, model.vars.state[i], stl.t_sym + model.dt))
        return stl.LinEq(terms, "=", 0)
    
    dyn_constrs = (row_to_stl(i, row) for i, row in enumerate(zip(A, B, C)))
    return stl.alw(stl.andf(*dyn_constrs), lo=0, hi=model.N)



def dyn_stl_to_matrix(dyn, model):
    """TODO: cleanup"""
    # TODO: initialize A, B, C
    dyn = set_time(dt=model.dt, phi=dyn)
    all_terms = {l.terms for l in stl.utils.lineq_lens(dyn).get_all()}
    rows = len(all_terms)
    A = np.zeros((rows, len(model.vars.state)))
    B = np.zeros((rows, len(model.vars.input)))
    C = np.zeros((rows, len(model.vars.env)))
    for i, terms in enumerate(all_terms):
        for coeff, name, _ in terms:
            if name in model.vars.state:
                A[i, model.vars.state.index(name)] = coeff
            elif name in model.vars.input:
                B[i, model.vars.input.index(name)] = coeff
            else:
                C[i, model.vars.env.index(name)] = coeff
    return A, B, C


def lookup_matrix(g, key, states, syms):
    N, M = len(states), len(syms)
    if key in g:
        data = g[key]
        path = data if pathlib.Path(data).exists() else io.BytesIO(
            data.encode('utf-8'))
        return np.genfromtxt(path).reshape((N, M))
    else:
        return np.zeros((N, M))


def parse_dynamics(g, model):
    # TODO
    if isinstance(g, dict):
        # TODO: assert shapes
        states = model.vars.state
        A = lookup_matrix(g, "A", states, states)
        A = np.eye(len(states)) + model.dt*A
        B = model.dt*lookup_matrix(g, "B", states, model.vars.input)
        C = model.dt*lookup_matrix(g, "C", states, model.vars.env)
        dyn = matrix_to_dyn_stl(A, B, C, model)
    else:
        dyn = stl.andf(*[stl.parse(entry['stl'], H=model.N) for entry in g])
        A, B, C = dyn_stl_to_matrix(dyn, model)
        
    Lu = dynamics_lipschitz(A, B, model.N)
    Lw = dynamics_lipschitz(A, C, model.N)
    return dyn, Lu, Lw


def parse_specs(g, model):
    # Parse Specs and Meta
    # TODO: Implement context
    spec_types = ["spec", "learned", "context"]
    spec_map = {k: stl.TOP for k in spec_types}
    pri_map = {}
    name_map = {}
    for kind in spec_types:
        for spec in g[kind]:
            p = stl.parse(spec['stl'], H=model.N)
            name_map[p] = spec.get('name')
            pri_map[p] = spec.get('pri')
            spec_map[kind] &= p

    dyn, dxdu, dxdw = parse_dynamics(g["dyn"], model)
    spec_map['dyn'] = dyn
    import ipdb; ipdb.set_trace()
    spec = Specs(**spec_map)
    drdx = stl.utils.linear_stl_lipschitz((~spec.env) | spec.sys)
    meta = Meta(pri_map, name_map, dxdu, dxdw, drdx)

    return spec, meta


def from_yaml(path) -> Game:
    if isinstance(path, (str, pathlib.Path)):
        with pathlib.Path(path).open("r") as f:
            g = defaultdict(list, yaml.load(f))
    else:
        g = defaultdict(list, yaml.load(f))

    model = parse_model(g)
    spec, meta = parse_specs(g, model)

    return Game(spec=spec, model=model, meta=meta)
