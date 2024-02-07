import os
import sys
import numpy as np
# import ExoPlex.make_grids as make_grids
# import ExoPlex.functions as functions
# hack to allow scripts to be placed in subdirectories next to ExoPlex:
if not os.path.exists('ExoPlex') and os.path.exists('../ExoPlex'):
    sys.path.insert(1, os.path.abspath('..'))

if __name__ == "__main__":
    # run here
    pass
