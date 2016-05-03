from __future__ import division

from math import ceil
from itertools import product, chain, starmap
from operator import mul
import operator

from singledispatch import singledispatch
from funcy import pairwise, mapcat
import gurobipy as gpy

import stl

M = 10000 # TODO
eps = 0.001 # TODO

def get_var(x, t, model, var_map):
    if x in var_map and t in var_map[x]:
        return var_map[x][t]
    if x not in var_map:
        var_map[x] = {}

    vtype = gpy.GRB.BINARY if isinstance(x, stl.Pred) else gpy.GRB.CONTINUOUS
    prefix = "z" if isinstance(x, stl.Pred) else "Z"
    i = len(var_map)
    name = "{}{}_{}".format(prefix, i, t)
    var_map[x][t] = model.addVar(vtype=vtype, name=name)
    model.update()
    return var_map[x][t]


def make_var_map(phi, model, steps, problem):
    var_map = {}

    n = problem['params']['num_vars']
    n_sys = problem['params']['num_sys_inputs']
    n_env = problem['params']['num_env_inputs']

    # Add state, input, and env vars
    for prefix, num in [('x', n), ('u', n_sys), ('w', n_env)]:
        var_map[prefix] = {(i, t): model.addVar(
            vtype=gpy.GRB.CONTINUOUS, name="{}{}_{}".format(prefix, i, t))
                           for i, t in product(range(num), range(steps))}

    model.update()

    return var_map

@singledispatch
def encode(problem):
    """STL -> MILP"""

    sys = reduce(stl.And, problem['sys'])
    env = reduce(stl.And, problem['env'], [])
    phi = stl.Or(stl.Neg(env), sys) if env else sys
        
    H = ceil(problem['params']['time_horizon'])
    dt = problem['params']['dt']
    steps = int(ceil(H / dt))

    model = gpy.Model("milp")
    var_map = make_var_map(phi, model, steps, problem)
    
    # encode STL constraints
    encode(phi, model, var_map, 0, H, dt)

    # add input bounds u in [0, 1]
    for u in chain(var_map['u'].values(), var_map['w'].values()):
        model.addConstr(u <= 1)
        model.addConstr(u >= 0)

    # TODO: Clean up system evolution
    n = problem['params']['num_vars']
    n_sys = problem['params']['num_sys_inputs']
    n_env = problem['params']['num_env_inputs']
    inputs = lambda t: [var_map['u'][i, t] for i in range(n_sys)] + [var_map['w'][i, t] for i in range(n_env)]
    state = lambda t: [var_map['x'][i, t] for i in range(n)]
    dot = lambda x, y: sum(starmap(mul, zip(x, y)))
    A, B = problem['state_space']['A'], problem['state_space']['B']
    for t in range(steps-1):
        for i, (A_i, B_i) in enumerate(zip(A, B)):
            model.addConstr(var_map['x'][i, t+1] == dot(A_i, state(t)) + dot(B_i, inputs(t)))

    # TODO: initial states
    for psi in problem['init']:
        x = var_map['x'][psi.lit, 0]
        const = psi.const
        model.addConstr(x == const)

    stl_vars =  mapcat(lambda x: var_map.get(x).values(), stl.walk(phi))
    model.setObjective(sum(stl_vars), gpy.GRB.MAXIMIZE)

    model.update()
    model.write('foo.lp')

@encode.register(stl.Pred)
def _(psi, model, var_map, t, H, dt):
    const = psi.const
    x = var_map['x'][psi.lit, t]
    z_t = get_var(psi, t, model, var_map)
    if psi.op in ("<", "<=", "="):
        model.addConstr(const - x <= M*z_t - eps)
        model.addConstr(x - const <= M*(1 - z_t) - eps)

    if psi.op in (">", ">=", "="):
        model.addConstr(x - const <= M*z_t - eps)
        model.addConstr(const - x <= M*(1 - z_t) - eps)


@encode.register(stl.Or)
def _(psi, model, var_map, t, H, dt):
    encode_bool_op(psi, model, var_map, t, H, dt, False)


@encode.register(stl.And)
def _(psi, model, var_map, t, H, dt):
    encode_bool_op(psi, model, var_map, t, H, dt, True)

    
def encode_bool_op(psi, model, var_map, t, H, dt, or_flag):
    z_psi = get_var(psi, t, model, var_map)
    elems = [get_var(psi.left, t, model, var_map), 
             get_var(psi.right, t, model, var_map)]

    encode_op(z_psi, elems, model, or_flag=or_flag)

    encode(psi.left, model, var_map, t, H, dt)
    encode(psi.right, model, var_map, t, H, dt)


@encode.register(stl.F)
def _(psi, model, var_map, t, H, dt):
    encode_temp_op(psi, model, var_map, t, H, dt, or_flag=True)


@encode.register(stl.G)
def _(psi, model, var_map, t, H, dt):
    encode_temp_op(psi, model, var_map, t, H, dt, or_flag=False)


def encode_temp_op(psi, model, var_map, t, H, dt, or_flag=False):
    z_psi = get_var(psi, t, model, var_map)
    a, b = psi.interval.lower, psi.interval.upper
    f = lambda x: int(ceil(x / dt))
    a, b = f(min(t + a, H)), f(min(t + b, H))
    
    elems = [get_var(psi.arg, i, model, var_map) for i in range(a, b)]
    encode_op(z_psi, elems, model, or_flag=or_flag)

    for i in range(a, b + 1):
        encode(psi.arg, model, var_map, t+i, H, dt)    


def encode_op(z_psi, elems, model, or_flag=False):
    z_phi_total = sum(elems)

    if or_flag:
        rel = operator.ge
        lhs = z_phi_total
    else: # AND
        rel = operator.le
        lhs = 1 - len(elems) + z_phi_total

    for e in elems:
        model.addConstr(rel(z_psi, e))
    model.addConstr(rel(lhs, z_psi))


@encode.register(stl.FG)
def _(psi, model, var_map, t, H, dt):
    raise NotImplementedError


@encode.register(stl.GF)
def _(psi, model, var_map, t, H, dt):
    raise NotImplementedError


@encode.register(stl.Neg)
def _(psi, model, var_map, t, H, dt):
    z_psi = get_var(psi, t, model, var_map)
    z_phi = get_var(psi.arg, t, model, var_map)
    model.addConstr(z_psi == 1 - z_phi)

    encode(psi.arg, model, var_map, t, H, dt)
