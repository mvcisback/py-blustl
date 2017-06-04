@0x850c57cbc8d1ab83;

struct Game {
    specs @0 :Specs;
    model @1 :Model;
    meta @2 :MetaData;

    struct Specs {
        objective @0 :Spec;
        dynamics @1 :List(Spec);

        # Learned Lemmas
        learned @2 :List(Spec);

        struct Spec {
            name @0 :Text;
            text @1 :Text;
            priority @2 :UInt64;
        }
    }

    struct Model {
        dt @0 :Float64;
        horizon @1 :Float64;

        state @2 :List(Var);
        inputs @3 :List(Var);
        environmentInput @4 :List(Var);
        currentTimeStep @5 :UInt64;

        struct Var {
           name @0 :Text;
           lowerBound @1 :Float64;
           upperBound @2 :Float64;
        }
    }

    struct MetaData {
        # Lipschitz Bounds if precomputed.
        dxdu @0 :Float64;
        dxdw @1 :Float64;
        drdx @2 :Float64;

        # Allow for unforeseen payloads.
        other @3 :Data;
    }
}