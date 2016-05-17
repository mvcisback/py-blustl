from enum import Enum

Kind = Enum(
    "Kind",
    [    
        "INIT",
        "DYNAMICS",
        "FIXED_INPUT",
        "SYS_INPUT_UPPER",
        "SYS_INPUT_LOWER",
        "ENV_INPUT_UPPER",
        "ENV_INPUT_LOWER",
        "ASSERT_FEASIBLE",
        "PRED_UPPER",
        "PRED_LOWER",
        "NEG",
        "F",
        "F_TOTAL",
        "G",
        "G_TOTAL",
        "OR",
        "OR_TOTAL",
        "AND",
        "AND_TOTAL",
    ]
)

class Category(Enum):
    Real = "Continuous"
    Bool = "Binary"
    Int = "Integer"
