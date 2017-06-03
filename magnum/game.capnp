@0x850c57cbc8d1ab83;

struct Game {
    specs @0 :Specs;
    model @1 :Model;
    meta @2 :MetaData;

    struct Specs {
        objectives @0 :List(Spec);
        dynamics @1 :List(Spec);

        # Learned Lemmas
        learned @2 :List(Spec);

        # AP -> Spec Association
        context @3 :List(Spec);

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
        input @3 :List(Var);
        environment_input @4 :List(Var);
        current_time_step @5 :UInt64;

        struct Var {
           name @0 :Text;
           low_bound @1 :Float64;
           upper_bound @1 :Float64;
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