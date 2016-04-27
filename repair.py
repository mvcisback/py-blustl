import stl

REPLACE_OP = {stl.F: stl.G, stl.G: stl.F}

def temporal_weaken(phi):
    """
    G -> FG -> GF -> F
    phi and psi -> phi
    """
    if not isinstance(phi, stl.Path_STL):
        raise NotImplemented

    phi2 = phi.arg
    i1 = phi.interval
    if isinstance(arg, stl.PATH_STL):
        i2 = phi2.interval
        phi3 = phi2.arg
        phi2 = REPLACE_OP[type(phi2)](phi3, i2)
    # TODO: actually implement switch shown above...
    
    return None
