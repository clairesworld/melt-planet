""" Calculate radial density distribution - start from running BurnMan """

import numpy as np
import burnman
from burnman import Mineral, PerplexMaterial, Composite, Layer, Planet
from burnman import minerals
from burnman import Composition
from burnman import BoundaryLayerPerturbation
from burnman.tools.chemistry import formula_mass
import warnings
import matplotlib.pyplot as plt

earth_mass = 5.972e24
earth_moment_of_inertia_factor = 0.3307


def make_test_earth():
    planet_radius = 6371.e3
    surface_temperature = 300.

    # ~~~~~~~~~~~~~~~~~~~~ CORE ~~~~~~~~~~~~~~~~~~~

    # Compositions from midpoints of Hirose et al. (2021), ignoring carbon and hydrogen
    inner_core_composition = Composition({'Fe': 94.4, 'Ni': 5., 'Si': 0.55, 'O': 0.05}, 'weight')
    outer_core_composition = Composition({'Fe': 90., 'Ni': 5., 'Si': 2., 'O': 3.}, 'weight')

    for c in [inner_core_composition, outer_core_composition]:
        c.renormalize('atomic', 'total', 1.)

    inner_core_elemental_composition = dict(inner_core_composition.atomic_composition)
    outer_core_elemental_composition = dict(outer_core_composition.atomic_composition)
    inner_core_molar_mass = formula_mass(inner_core_elemental_composition)
    outer_core_molar_mass = formula_mass(outer_core_elemental_composition)

    # Set up inner core - assume it's adiabatic which it probably isn't but shouldn't matter for structure

    icb_radius = 1220.e3
    inner_core = Layer('inner core', radii=np.linspace(0., icb_radius, 21))

    hcp_iron = minerals.SE_2015.hcp_iron()
    params = hcp_iron.params

    params['name'] = 'modified solid iron'
    params['formula'] = inner_core_elemental_composition
    params['molar_mass'] = inner_core_molar_mass
    delta_V = 2.0e-7

    inner_core_material = Mineral(params=params,
                                  property_modifiers=[['linear',
                                                       {'delta_E': 0.,
                                                        'delta_S': 0.,
                                                        'delta_V': delta_V}]])

    # check that the new inner core material does what we expect:
    hcp_iron.set_state(200.e9, 4000.)
    inner_core_material.set_state(200.e9, 4000.)
    assert np.abs(delta_V - (inner_core_material.V - hcp_iron.V)) < 1.e-12

    inner_core.set_material(inner_core_material)

    inner_core.set_temperature_mode('adiabatic')

    # Set up outer core (also assume adiabatic)

    cmb_radius = 3480.e3
    outer_core = Layer('outer core', radii=np.linspace(icb_radius, cmb_radius, 21))

    liq_iron = minerals.SE_2015.liquid_iron()
    params = liq_iron.params

    params['name'] = 'modified liquid iron'
    params['formula'] = outer_core_elemental_composition
    params['molar_mass'] = outer_core_molar_mass
    delta_V = -2.3e-7
    outer_core_material = Mineral(params=params,
                                  property_modifiers=[['linear',
                                                       {'delta_E': 0.,
                                                        'delta_S': 0.,
                                                        'delta_V': delta_V}]])

    # check that the new inner core material does what we expect:
    liq_iron.set_state(200.e9, 4000.)
    outer_core_material.set_state(200.e9, 4000.)
    assert np.abs(delta_V - (outer_core_material.V - liq_iron.V)) < 1.e-12

    outer_core.set_material(outer_core_material)

    outer_core.set_temperature_mode('adiabatic')

    # ~~~~~~~~~~~~~~~~~~~~~~~~ MANTLE ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # assume a single covecting mantle layer with a Perple_X input table (so equilibrium stabilities are asserted)
    # because BurnMan composite isn't fast enough to calculate equilibrium mineral properties on the fly

    lab_radius = 6171.e3 # 200 km thick lithosphere
    lab_temperature = 1550.

    convecting_mantle_radii = np.linspace(cmb_radius, lab_radius, 101)
    convecting_mantle = Layer('convecting mantle', radii=convecting_mantle_radii)

    # Import a low resolution PerpleX data table.
    fname = '/home/claire/Works/burnman/tutorial/data/pyrolite_perplex_table_lo_res.dat'
    pyrolite = PerplexMaterial(fname, name='pyrolite')
    convecting_mantle.set_material(pyrolite)

    # Here we add a thermal boundary layer perturbation, assuming that the
    # lower mantle has a Rayleigh number of 1.e7, and that the basal thermal
    # boundary layer has a temperature jump of 840 K and the top
    # boundary layer has a temperature jump of 60 K.
    tbl_perturbation = BoundaryLayerPerturbation(radius_bottom=cmb_radius,
                                                 radius_top=lab_radius,
                                                 rayleigh_number=1.e7,
                                                 temperature_change=900.,
                                                 boundary_layer_ratio=60./900.)

    # Onto this perturbation, we add a linear superadiabaticity term according
    # to Anderson (he settled on 200 K over the lower mantle)
    dT_superadiabatic = 300.*(convecting_mantle_radii - convecting_mantle_radii[-1])/(convecting_mantle_radii[0] - convecting_mantle_radii[-1])

    convecting_mantle_tbl = (tbl_perturbation.temperature(convecting_mantle_radii)
                             + dT_superadiabatic)

    convecting_mantle.set_temperature_mode('perturbed-adiabatic',
                                           temperatures=convecting_mantle_tbl)

    # Initialise lithosphere with user-defined conductive gradient

    moho_radius = 6341.e3
    moho_temperature = 620.

    dunite = minerals.SLB_2011.mg_fe_olivine(molar_fractions=[0.92, 0.08])
    lithospheric_mantle = Layer('lithospheric mantle',
                                radii=np.linspace(lab_radius, moho_radius, 31))
    lithospheric_mantle.set_material(dunite)
    lithospheric_mantle.set_temperature_mode('user-defined',
                                             np.linspace(lab_temperature,
                                                         moho_temperature, 31))

    # Finally, we assume the crust has the density of andesine ~ 40% anorthite

    andesine = minerals.SLB_2011.plagioclase(molar_fractions=[0.4, 0.6])
    crust = Layer('crust', radii=np.linspace(moho_radius, planet_radius, 11))
    crust.set_material(andesine)
    crust.set_temperature_mode('user-defined',
                               np.linspace(moho_temperature,
                                           surface_temperature, 11))

    # Now build planet
    planet_zog = Planet('Planet Zog',
                        [inner_core, outer_core,
                         convecting_mantle, lithospheric_mantle,
                         crust], verbose=True)
    planet_zog.make()

    # Output
    print(f'mass = {planet_zog.mass:.3e} (Earth = {earth_mass:.3e})')
    print(f'moment of inertia factor= {planet_zog.moment_of_inertia_factor:.4f} '
          f'(Earth = {earth_moment_of_inertia_factor:.4f})')

    print('Layer mass fractions:')
    for layer in planet_zog.layers:
        print(f'{layer.name}: {layer.mass / planet_zog.mass:.3f}')

    print('size of inner + outer core:', len(inner_core.radii) + len(outer_core.radii))

    return planet_zog



def plot_structure_full(planet_zog):
    # Now let's plot everything up

    # Optional prettier plotting
    plt.style.use('ggplot')

    # Let's get PREM to compare everything to as we are trying
    # to imitate Earth
    prem = burnman.seismic.PREM()
    premradii = 6371.e3 - prem.internal_depth_list()

    with warnings.catch_warnings(record=True) as w:
        eval = prem.evaluate(['density', 'pressure', 'gravity', 'v_s', 'v_p'])
        premdensity, prempressure, premgravity, premvs, premvp = eval
        print(w[-1].message)

    figure = plt.figure(figsize=(10, 12))
    figure.suptitle(
        '{0} has a mass {1:.3f} times that of Earth,\n'
        'has an average density of {2:.1f} kg/m$^3$,\n'
        'and a moment of inertia factor of {3:.4f}'.format(
            planet_zog.name,
            planet_zog.mass / 5.97e24,
            planet_zog.average_density,
            planet_zog.moment_of_inertia_factor),
        fontsize=20)

    ax = [figure.add_subplot(4, 1, i) for i in range(1, 5)]

    ax[0].plot(planet_zog.radii / 1.e3, planet_zog.density / 1.e3, 'k', linewidth=2.,
               label=planet_zog.name)
    ax[0].plot(premradii / 1.e3, premdensity / 1.e3, '--k', linewidth=1.,
               label='PREM')
    ax[0].set_ylim(0., (max(planet_zog.density) / 1.e3) + 1.)
    ax[0].set_ylabel('Density ($\cdot 10^3$ kg/m$^3$)')
    ax[0].legend()

    # Make a subplot showing the calculated pressure profile
    ax[1].plot(planet_zog.radii / 1.e3, planet_zog.pressure / 1.e9, 'b', linewidth=2.)
    ax[1].plot(premradii / 1.e3, prempressure / 1.e9, '--b', linewidth=1.)
    ax[1].set_ylim(0., (max(planet_zog.pressure) / 1e9) + 10.)
    ax[1].set_ylabel('Pressure (GPa)')

    # Make a subplot showing the calculated gravity profile
    ax[2].plot(planet_zog.radii / 1.e3, planet_zog.gravity, 'g', linewidth=2.)
    ax[2].plot(premradii / 1.e3, premgravity, '--g', linewidth=1.)
    ax[2].set_ylabel('Gravity (m/s$^2)$')
    ax[2].set_ylim(0., max(planet_zog.gravity) + 0.5)

    # Make a subplot showing the calculated temperature profile
    mask = planet_zog.temperature > 301.
    ax[3].plot(planet_zog.radii[mask] / 1.e3,
               planet_zog.temperature[mask], 'r', linewidth=2.)
    ax[3].set_ylabel('Temperature ($K$)')
    ax[3].set_xlabel('Radius (km)')
    ax[3].set_ylim(0., 7000)  #max(planet_zog.temperature) + 100)

    for i in range(3):
        ax[i].set_xticklabels([])
    for i in range(4):
        ax[i].set_xlim(0., max(planet_zog.radii) / 1.e3)

    plt.show()


pl = make_test_earth()
# plot_structure_full(pl)
print(pl.temperature)
print(pl.radii)