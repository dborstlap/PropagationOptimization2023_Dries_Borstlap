# QUESTIONS
# Why suddenly more error when time step 0.5 and 1.0 s
#                            and also at 0.1 and 0.2   ==> probably because lagrange interpolator at end position is wack
# when error based on /2, it is straight line. When *2 it is jagged. Why?    ==> probably because lagrange interpolator at end position is wack

# what to plot for Q1b? Because 3d orbit makes no sense

# is there a line limit for rest of Q1?

# is there a handy overview of which coefficient sets go together with which fixed_step_size/variable_step_size/bulirsch_stoer_type/...

# is 0.1m benchmark sufficient

# NOTE: not start at pos=0 because it would be singularity in spherical elements

# what is fixed-order integrator?

# how to remove edges of the interpolator

"""
Copyright (c) 2010-2022, Delft University of Technology
All rights reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.

AE4866 Propagation and Optimization in Astrodynamics
Lunar Ascent
First name: ***COMPLETE HERE***
Last name: ***COMPLETE HERE***
Student number: ***COMPLETE HERE***

This module computes the dynamics of a Lunar ascent vehicle, according to a simple thrust guidance law.  This file propagates the dynamics
using a variety of integrator and propagator settings. For each run, the differences w.r.t. a benchmark propagation are
computed, providing a proxy for setting quality. The benchmark settings are currently defined semi-randomly, and are to be
analyzed/modified.

The propagtion starts with a small velocity close to the surface of the Moon, and an initial flight path angle of 90
degrees. Making (small) adjustments to this initial state is permitted if properly argued in the report.

The propagation is terminated as soon as one of the following conditions is met:

- Altitude > 100 km
- Altitude < 0 km
- Propagation time > 3600 s
- Vehicle mass < 2250 kg

This propagation assumes only point mass gravity by the Moon and thrust acceleration of the vehicle. Both the
translational dynamics and mass of the vehicle are propagated, using a fixed specific impulse.

The thrust is computed based on a constant thrust magnitude, and a variable thrust direction. The trust direction is defined
on a set of 5 nodes, spread evenly in time. At each node, a thrust angle theta is defined, which gives the angle between
the -z and y angles in the ascent vehicle's vertical frame (see Mooij, 1994, "The motion of a vehicle in a planetary
atmosphere" ). Between the nodes, the thrust is linearly interpolated. If the propagation goes beyond the bounds of
the nodes, the boundary value is used. The thrust profile is parameterized by the values of the vector thrust_parameters.
The thrust guidance is implemented in the LunarAscentThrustGuidance class in the LunarAscentUtilities.py file.

The entries of the vector 'thrust_parameters' contains the following:
- Entry 0: Constant thrust magnitude
- Entry 1: Constant spacing in time between nodes
- Entry 2-6: Thrust angle theta, at nodes 1-5 (in order)

Details on the outputs written by this file can be found:
- benchmark data: comments for 'generateBenchmarks' function
- results for integrator/propagator variations: comments under "RUN SIMULATION FOR VARIOUS SETTINGS"

Frequent warnings and/or errors that might pop up:
* One frequent warning could be the following (mock values):
    "Warning in interpolator, requesting data point outside of boundaries, requested data at 7008 but limit values are
    0 and 7002, applying extrapolation instead."
It can happen that the benchmark ends earlier than the regular simulation, due to the smaller step size. Therefore,
the code will be forced to extrapolate the benchmark states (or dependent variables) to compare them to the
simulation output, producing a warning. This warning can be deactivated by forcing the interpolator to use the boundary
value instead of extrapolating (extrapolation is the default behavior). This can be done by setting:

    interpolator_settings = interpolators.lagrange_interpolation(
        8, boundary_interpolation = interpolators.extrapolate_at_boundary)

* One frequent error could be the following:
    "Error, propagation terminated at t=4454.723896, returning propagation data up to current time."
    This means that an error occurred with the given settings. Typically, this implies that the integrator/propagator
    combination is not feasible. It is part of the assignment to figure out why this happens.

* One frequent error can be one of:
    "Error in RKF integrator, step size is NaN"
    "Error in ABM integrator, step size is NaN"
    "Error in BS integrator, step size is NaN"

This means that a variable time-step integrator wanting to take a NaN time step. In such cases, the selected
integrator settings are unsuitable for the problem you are considering.

NOTE: When any of the above errors occur, the propagation results up to the point of the crash can still be extracted
as normal. It can be checked whether any issues have occured by using the function

dynamics_simulator.integration_completed_successfully

which returns a boolean (false if any issues have occured)

* A frequent issue can be that a simulation with certain settings runs for too long (for instance if the time steo
becomes excessively small). To prevent this, you can add an additional termination setting (on top of the existing ones!)

    cpu_tim_termination_settings = propagation_setup.propagator.cpu_time_termination(
        maximum_cpu_time )

where maximum_cpu_time is a varaiable (float) denoting the maximum time in seconds that your simulation is allowed to
run. If the simulation runs longer, it will terminate, and return the propagation results up to that point.

* Finally, if the following error occurs, you can NOT extract the results up to the point of the crash. Instead,
the program will immediately terminate

    SPICE(DAFNEGADDR) --

    Negative value for BEGIN address: -214731446

This means that a state is extracted from Spice at a time equal to NaN. Typically, this is indicative of a
variable time-step integrator wanting to take a NaN time step, and the issue not being caught by Tudat.
In such cases, the selected integrator settings are unsuitable for the problem you are considering.
"""

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np
import os

# Tudatpy imports
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.math import interpolators

# Problem-specific imports
import LunarAscentUtilities as Util

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################


# Load spice kernels
spice_interface.load_standard_kernels()
# NOTE TO STUDENTS: INPUT YOUR PARAMETER SET HERE, FROM THE INPUT FILES
# ON BRIGHTSPACE, FOR YOUR SPECIFIC STUDENT NUMBER

thrust_parameters = [17869.1842977423, 21.5312002995, 0.0895461222, -0.3786714207, 0.4978693228, -0.2725262092, -1.132938021]

# Choose whether benchmark is run
use_benchmark = True
run_integrator_analysis = True


# Get path of current directory
current_dir = os.path.dirname(__file__)

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Set simulation start epoch
simulation_start_epoch = 0.0  # s
# Vehicle settings
vehicle_mass = 4.7E3  # kg
vehicle_dry_mass = 2.25E3  # kg
constant_specific_impulse = 311.0  # s
# Fixed simulation termination settings
maximum_duration = constants.JULIAN_DAY  # s
termination_altitude = 100.0E3  # m

###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

# Define settings for celestial bodies
bodies_to_create = ['Moon']
# Define coordinate system
global_frame_origin = 'Moon'
global_frame_orientation = 'ECLIPJ2000'

# Create body settings
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)
# Create bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# Create vehicle object and add it to the existing system of bodies
bodies.create_empty_body('Vehicle')
# Set mass of vehicle
bodies.get_body('Vehicle').mass = vehicle_mass

# Create thrust model, with dummy settings, to be overridden when processing the thrust parameters
thrust_magnitude_settings = (
    propagation_setup.thrust.constant_thrust_magnitude( thrust_magnitude=0.0, specific_impulse=constant_specific_impulse ) )
environment_setup.add_engine_model(
    'Vehicle', 'MainEngine', thrust_magnitude_settings, bodies )
environment_setup.add_rotation_model(
    bodies, 'Vehicle', environment_setup.rotation_model.custom_inertial_direction_based(
        lambda time : np.array([1,0,0] ), global_frame_orientation, 'VehicleFixed' ) )
###########################################################################
# CREATE PROPAGATOR SETTINGS ##############################################
###########################################################################

# Retrieve termination settings
termination_settings = Util.get_termination_settings(simulation_start_epoch,
                                                     maximum_duration,
                                                     termination_altitude,
                                                     vehicle_dry_mass)
# Retrieve dependent variables to save
dependent_variables_to_save = Util.get_dependent_variable_save_settings()

###########################################################################
# IF DESIRED, GENERATE BENCHMARK ##########################################
###########################################################################

if use_benchmark:

    time_step = 0.05

    first_benchmark_dependent_variable_history = {}
    with open("SimulationOutput/benchmarks/benchmark_1_dependent_variables.dat") as f:
        for line in f:
            (key, val1, val2, val3) = line.split()
            first_benchmark_dependent_variable_history[float(key)] = [float(val1), float(val2), float(val3)]

    first_benchmark_state_history = {}
    with open("SimulationOutput/benchmarks/benchmark_1_states.dat") as f:
        for line in f:
            (key, val1, val2, val3, val4, val5, val6, val7) = line.split()
            first_benchmark_state_history[float(key)] = [float(val1), float(val2), float(val3), float(val4), float(val5), float(val6), float(val7)]

    # Define benchmark interpolator settings to make a comparison between the two benchmarks
    benchmark_interpolator_settings = interpolators.lagrange_interpolation(
        8,boundary_interpolation = interpolators.extrapolate_at_boundary)

    # make interpolators
    benchmark_state_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        first_benchmark_state_history, benchmark_interpolator_settings)
    benchmark_dependent_variable_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        first_benchmark_dependent_variable_history, benchmark_interpolator_settings)


    benchmark_keys = list(first_benchmark_state_history.keys())
    valid_interpolation_epochs = benchmark_keys[4:-4]
    valid_interpolation_range = [valid_interpolation_epochs[0], valid_interpolation_epochs[-1]]



    #### IF GENERATING NEW BENCHMARKS ######

    # benchmark_output_path = current_dir + '/SimulationOutput/benchmarks/'
    #
    # # Create propagator settings for benchmark (Cowell)
    # propagator_settings = Util.get_propagator_settings(
    #     thrust_parameters,
    #     bodies,
    #     simulation_start_epoch,
    #     vehicle_mass,
    #     termination_settings,
    #     dependent_variables_to_save)
    #
    # # for time_step in np.logspace(-6, 6, base=2, num=49):
    # for time_step in [0.05]:
    #     print()
    #     print('TIME STEP = ', time_step)
    #     print()
    #
    #     benchmark_step_size = time_step
    #     benchmark_list = Util.generate_benchmarks(benchmark_step_size,
    #                                               bodies,
    #                                               propagator_settings,
    #                                               True,
    #                                               benchmark_output_path)
    #
    #     # Extract benchmark states
    #     first_benchmark_state_history = benchmark_list[0]
    #     second_benchmark_state_history = benchmark_list[1]
    #
    #     # Create state interpolator for first benchmark
    #     benchmark_state_interpolator = interpolators.create_one_dimensional_vector_interpolator(
    #         first_benchmark_state_history,
    #         benchmark_interpolator_settings)
    #
    #
    #
    #     #####################
    #
    #     benchmark_keys = list(first_benchmark_state_history.keys())
    #     valid_interpolation_epochs = benchmark_keys[5:-5]
    #     valid_interpolation_range = [valid_interpolation_epochs[0], valid_interpolation_epochs[-1]]
    #
    #     # keys = list(benchmark_state_interpolator.keys())
    #     # keys_to_pop = keys[:4] + keys[-4:]
    #     # for key_to_pop in keys_to_pop:
    #     #     # first_benchmark_state_history.pop()
    #     #     del benchmark_state_interpolator[key_to_pop]
    #
    #     # Compare benchmark states, returning interpolator of the first benchmark, and writing difference to file
    #     benchmark_state_difference = Util.compare_benchmarks(first_benchmark_state_history,
    #                                                          second_benchmark_state_history,
    #                                                          benchmark_output_path,
    #                                                          'benchmarks_state_difference_'+str(time_step)+'.dat')
    #
    #     # Extract benchmark dependent variables, if present
    #     if are_dependent_variables_to_save:
    #         first_benchmark_dependent_variable_history = benchmark_list[2]
    #         second_benchmark_dependent_variable_history = benchmark_list[3]
    #
    #         # Create dependent variable interpolator for first benchmark
    #         benchmark_dependent_variable_interpolator = interpolators.create_one_dimensional_vector_interpolator(
    #             first_benchmark_dependent_variable_history,
    #             benchmark_interpolator_settings)
    #
    #         # Compare benchmark dependent variables, returning interpolator of the first benchmark, and writing difference
    #         benchmark_dependent_difference = Util.compare_benchmarks(first_benchmark_dependent_variable_history,
    #                                                                  second_benchmark_dependent_variable_history,
    #                                                                  benchmark_output_path,
    #                                                                  'benchmarks_dependent_variable_difference.dat')
    #
    #     Util.plot_compared_benchmarks('/SimulationOutput/benchmarks/benchmarks_state_difference.dat')

###########################################################################
# RUN SIMULATION FOR VARIOUS SETTINGS #####################################
###########################################################################
"""
Code below propagates states using each propagator (propagator_index=0..6), four multi-stage variable step-size
integrators (integrator_index j=0..3) and an RK4 integrator (j=4). For the variable-step integrators, 4 different
tolerances are used (step_size_index=0..3). For the RK4, 6 different step sizes are used (step_size_index=0..5),
see use of n_time_step_settings variable. See get_integrator_settings function for more details.

For each combination of i, j, and k, results are written to directory:
    LunarAscent/SimulationOutput/prop_i/int_j/setting_k/

Specifically:
     state_History.dat                                  Cartesian states as function of time
     dependent_variable_history.dat                     Dependent variables as function of time
     state_difference_wrt_benchmark.dat                 Difference of dependent variables w.r.t. benchmark
     dependent_variable_difference_wrt_benchmark.dat    Difference of states w.r.t. benchmark
     ancillary_simulation_info.dat                      Other information about the propagation (number of function
                                                        evaluations, etc...)

NOTE TO STUDENTS: THE NUMBER, TYPES, SETTINGS OF PROPAGATORS/INTEGRATORS/INTEGRATOR STEPS,TOLERANCES,ETC. SHOULD BE
MODIFIED FOR ASSIGNMENT 1.
"""
if run_integrator_analysis:

    n_propagators = 7
    n_integrators = 15
    n_step_sizes = 5

    # Loop over propagators
    for propagator_index in range(n_propagators):
        # Get current propagator, and define translational state propagation settings
        current_propagator = Util.get_propagator(propagator_index)

        # Define propagation settings
        current_propagator_settings = Util.get_propagator_settings(
            thrust_parameters,
            bodies,
            simulation_start_epoch,
            vehicle_mass,
            termination_settings,
            dependent_variables_to_save,
            current_propagator)

        if propagator_index == 0:
            # integrator_indexes = range(n_integrators)
            integrator_indexes = [1]
            # integrator_indexes = [12, 13, 14]
        else:
            integrator_indexes = [1] # the best one

        for integrator_index in integrator_indexes:

            # Loop over all tolerances / step sizes
            for step_size_index in range(n_step_sizes):
                print('Current run: \n propagator_index = ' + str(propagator_index) + '\n integrator_index = ' + str(integrator_index) + '\n step_size_index = ' + str(step_size_index))

                # Set output path
                output_path = current_dir + '/SimulationOutput/prop_' + str(propagator_index) + '/int_' + str(integrator_index) + '/step_size_' + str(step_size_index) + '/'

                # Create integrator settings
                current_integrator_settings = Util.get_integrator_settings(
                    propagator_index,
                    integrator_index,
                    step_size_index,
                    simulation_start_epoch)

                # further define propagator
                current_propagator_settings.integrator_settings = current_integrator_settings

                # Create Lunar Ascent Problem object
                dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, current_propagator_settings )


                ###########################################################################
                # OUTPUT OF SIMULATIOIN #####################################
                ###########################################################################

                # Retrieve propagated state and dependent variables
                state_history = dynamics_simulator.state_history
                unprocessed_state_history = dynamics_simulator.unprocessed_state_history
                dependent_variable_history = dynamics_simulator.dependent_variable_history

                # Get the number of function evaluations (for comparison of different integrators)
                function_evaluation_dict = dynamics_simulator.cumulative_number_of_function_evaluations
                n_function_evaluations = list(function_evaluation_dict.values())[-1]
                # Add it to a dictionary
                dict_to_write = {'Number of function evaluations': n_function_evaluations}
                # Check if the propagation was run successfully
                propagation_outcome = dynamics_simulator.integration_completed_successfully
                dict_to_write['Propagation run successfully'] = propagation_outcome
                # Note if results were written to files
                dict_to_write['Results written to file'] = True
                # Note if benchmark was run
                dict_to_write['Benchmark run'] = use_benchmark
                # Note if dependent variables were present
                dict_to_write['Dependent variables present'] = True

                # Save results to a file
                save2txt(state_history, 'state_history.dat', output_path)
                save2txt(unprocessed_state_history, 'unprocessed_state_history.dat', output_path)
                save2txt(dependent_variable_history, 'dependent_variable_history.dat', output_path)
                save2txt(dict_to_write, 'ancillary_simulation_info.txt',   output_path)

                ### BENCHMARK COMPARISON ####
                # Compare the simulation to the benchmarks and write differences to files
                if use_benchmark:
                    # Initialize containers
                    state_difference = dict()
                    # state_difference[0] = 0

                    # Loop over the propagated states and use the benchmark interpolators
                    # NOTE TO STUDENTS: it can happen that the benchmark ends earlier than the regular simulation, due to
                    # the shorter step size. Therefore, the following lines of code will be forced to extrapolate the
                    # benchmark states (or dependent variables), producing a warning. Be aware of it!

                    # state_interpolator_settings = interpolators.lagrange_interpolation(
                    #     2, boundary_interpolation=interpolators.extrapolate_at_boundary)
                    #
                    # state_interpolator = interpolators.create_one_dimensional_vector_interpolator(
                    #     state_history,
                    #     state_interpolator_settings)

                    for epoch in state_history.keys():
                        # print(epoch, ':::', type(epoch))
                        if epoch >= valid_interpolation_range[0] and epoch <= valid_interpolation_range[-1]:
                            state_difference[epoch] = state_history[epoch] - benchmark_state_interpolator.interpolate(epoch)


                    # Write differences with respect to the benchmarks to files
                    Util.write_with_exception(state_difference, file_name='state_difference_wrt_benchmark.dat', output_path=output_path)

                    # Do the same for dependent variables
                    dependent_difference = dict()
                    # Loop over the propagated dependent variables and use the benchmark interpolators
                    for epoch in dependent_variable_history.keys():
                        dependent_difference[epoch] = dependent_variable_history[epoch] - benchmark_dependent_variable_interpolator.interpolate(epoch)

                    # Write differences with respect to the benchmarks to files
                    Util.write_with_exception(dependent_difference, file_name='dependent_variable_difference_wrt_benchmark.dat',output_path=output_path)



    # Print the ancillary information
    print('\n### ANCILLARY SIMULATION INFORMATION ###')
    for (elem, (info, result)) in enumerate(dict_to_write.items()):
        if elem > 1:
            print(info + ': ' + str(result))
