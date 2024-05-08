import sys
sys.path
sys.path.append('/home/claire/Works/exo-top/')
import exotop.model_1D as bl



bl.evolve.build_planet(planet_kwargs, run_kwargs, solve_ODE=True)