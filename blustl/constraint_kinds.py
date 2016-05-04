from enum import Enum, unique


@unique
class Kind(Enum):
    INIT = 1
    DYNAMICS = 2
    SYS_INPUT_UPPER = 3
    SYS_INPUT_LOWER = 4
    ENV_INPUT_UPPER = 5
    ENV_INPUT_LOWER = 6
    ASSERT_FEASIBLE = 7
    PRED_UPPER = 8
    PRED_LOWER = 9
    NEG = 10
    F = 11
    F_TOTAL = 12
    G = 13
    G_TOTAL = 14
    OR = 15
    OR_TOTAL = 16
    AND = 17
    AND_TOTAL = 18
