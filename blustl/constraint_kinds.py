from enum import Enum

UNREPAIRABLE = {
    "INIT",
    "DYNAMICS", 
    "FIXED_INPUT",
    "SYS_INPUT_UPPER",
    "SYS_INPUT_LOWER",
    "ENV_INPUT_UPPER",
    "ENV_INPUT_LOWER",
    "ASSERT_FEASIBLE",
}

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
        "PRED_EQ",
        "NEG",
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
