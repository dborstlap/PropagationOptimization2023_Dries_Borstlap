'''
Copyright (c) 2010-2021, Delft University of Technology
All rights reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.

AE4866 Propagation and Optimization in Astrodynamics
Low Thrust
First name: ***COMPLETE HERE***
Last name: ***COMPLETE HERE***
Student number: ***COMPLETE HERE***

This module defines useful functions that will be called by the main script, where the optimization is executed.
'''

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np

# Tudatpy imports
import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.simulation import propagation_setup
from tudatpy.kernel.math import interpolators

# Problem-specific imports
import LowThrustProblem as LowThrust


###########################################################################
# USEFUL FUNCTIONS ########################################################
###########################################################################


def get_termination_settings(trajectory_parameters,
                             minimum_mars_distance: float,
                             time_buffer: float) \
        -> tudatpy.kernel.simulation.propagation_setup.propagator.PropagationTerminationSettings:
    """
    Get the termination settings for the simulation.

    Termination settings currently include:
    - simulation time (propagation stops if it is greater than the one provided by the hodographic trajectory)
    - distance to Mars (propagation stops if the relative distance is lower than the target distance)

    Parameters
    ----------
    trajectory_parameters : list[floats]
        List of trajectory parameters.
    minimum_mars_distance : float
        Minimum distance from Mars at which the propagation stops.
    time_buffer : float
        Time interval between the simulation start epoch and the beginning of the hodographic trajectory.

    Returns
    -------
    hybrid_termination_settings : tudatpy.kernel.simulation.propagation_setup.propagator.PropagationTerminationSettings
        Propagation termination settings object.
    """
    # Create single PropagationTerminationSettings objects
    # Time
    final_time = LowThrust.get_trajectory_final_time(trajectory_parameters,
                                                     time_buffer)
    time_termination_settings = propagation_setup.propagator.time_termination(
        final_time,
        terminate_exactly_on_final_condition=False)
    # Altitude
    relative_distance_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.relative_distance('Vehicle', 'Mars'),
        limit_value=minimum_mars_distance,
        use_as_lower_limit=True,
        terminate_exactly_on_final_condition=False)
    # Define list of termination settings
    termination_settings_list = [time_termination_settings,
                                 relative_distance_termination_settings]
    # Create termination settings object
    hybrid_termination_settings = propagation_setup.propagator.hybrid_termination(termination_settings_list,
                                                                                  fulfill_single_condition=True)
    return hybrid_termination_settings


# NOTE TO STUDENTS: this function can be modified to save more/less dependent variables.
def get_dependent_variable_save_settings() -> list:
    """
    Retrieves the dependent variables to save.

    Currently, the dependent variables saved include the relative distance between the spacecraft and:
    - Earth
    - the Sun
    - Mars

    Parameters
    ----------
    none

    Returns
    -------
    dependent_variables_to_save : list[tudatpy.kernel.simulation.propagation_setup.dependent_variable]
        List of dependent variables to save.
    """
    dependent_variables_to_save = [propagation_setup.dependent_variable.relative_distance('Vehicle', 'Earth'),
                                   propagation_setup.dependent_variable.relative_distance('Vehicle', 'Sun'),
                                   propagation_setup.dependent_variable.relative_distance('Vehicle', 'Mars')]
    return dependent_variables_to_save


# NOTE TO STUDENTS: THIS FUNCTION SHOULD BE EXTENDED TO USE MORE INTEGRATORS FOR ASSIGNMENT 1.
def get_integrator_settings(propagator_index: int,
                            integrator_index: int,
                            settings_index: int,
                            simulation_start_epoch: float) \
        -> tudatpy.kernel.simulation.propagation_setup.integrator.IntegratorSettings:
    """

    Retrieves the integrator settings.

    It selects a combination of integrator to be used (first argument) and
    the related setting (tolerance for variable step size integrators
    or step size for fixed step size integrators). The code, as provided, runs the following:
    - if j=0,1,2,3: a variable-step-size, multi-stage integrator is used (see multiStageTypes list for specific type),
                     with tolerances 10^(-10+*k)
    - if j=4      : a fixed-step-size RK4 integrator is used, with step-size 7200*2^(k)

    Parameters
    ----------
    propagator_index : int
        Index that selects the propagator type (currently not used).
        NOTE TO STUDENTS: this argument can be used to select specific combinations of propagator and integrators
        (provided that the code is expanded).
    integrator_index : int
        Index that selects the integrator type as follows:
            0 -> RK4(5)
            1 -> RK5(6)
            2 -> RK7(8)
            3 -> RKDP7(8)
            4 -> RK4
    settings_index : int
        Index that selects the tolerance or the step size
        (depending on the integrator type).
    simulation_start_epoch : float
        Start of the simulation [s] with t=0 at J2000.

    Returns
    -------
    integrator_settings : tudatpy.kernel.simulation.propagation_setup.integrator.IntegratorSettings
        Integrator settings to be provided to the dynamics simulator.

    """
    # Define list of multi-stage integrators
    multi_stage_integrators = [propagation_setup.integrator.RKCoefficientSets.rkf_45,
                               propagation_setup.integrator.RKCoefficientSets.rkf_56,
                               propagation_setup.integrator.RKCoefficientSets.rkf_78,
                               propagation_setup.integrator.RKCoefficientSets.rkdp_87]

    # Use variable step-size integrator
    if integrator_index < 4:
        # Select variable-step integrator
        current_coefficient_set = multi_stage_integrators[integrator_index]
        # Compute current tolerance
        current_tolerance = 10.0 ** (-10.0 + settings_index)
        # Create integrator settings
        integrator = propagation_setup.integrator
        # Here (epsilon, inf) are set as respectively min and max step sizes
        # also note that the relative and absolute tolerances are the same value
        integrator_settings = integrator.runge_kutta_variable_step_size(simulation_start_epoch,
                                                                        1.0,
                                                                        current_coefficient_set,
                                                                        np.finfo(float).eps,
                                                                        np.inf,
                                                                        current_tolerance,
                                                                        current_tolerance)
    # Use fixed step-size integrator
    else:
        # Compute time step
        fixed_step_size = 7200.0 * 2.0 ** settings_index
        # Create integrator settings
        integrator = propagation_setup.integrator
        integrator_settings = integrator.runge_kutta_4(simulation_start_epoch,
                                                       fixed_step_size)
    return integrator_settings


# NOTE TO STUDENTS: THIS FUNCTION CAN BE EXTENDED TO GENERATE A MORE ROBUST BENCHMARK (USING MORE THAN 2 RUNS)
def generate_benchmarks(benchmark_step_size: float,
			simulation_start_epoch: float,
                        specific_impulse: float,
                        minimum_mars_distance: float,
                        time_buffer: float,
                        bodies: tudatpy.kernel.simulation.environment_setup.SystemOfBodies,
                        benchmark_propagator_settings:
                        tudatpy.kernel.simulation.propagation_setup.propagator.MultiTypePropagatorSettings,
                        trajectory_parameters: list,
                        are_dependent_variables_present: bool,
                        output_path: str = None):
    """
    Function to generate to accurate benchmarks.

    This function runs two propagations with two different integrator settings that serve as benchmarks for
    the nominal runs. The state and dependent variable history for both benchmarks are returned and, if desired, 
    they are also written to files (to the directory ./SimulationOutput/benchmarks/) in the following way:
    * benchmark_1_states.dat, benchmark_2_states.dat
        The numerically propagated states from the two benchmarks.
    * benchmark_1_dependent_variables.dat, benchmark_2_dependent_variables.dat
        The dependent variables from the two benchmarks.

    Parameters
    ----------
    simulation_start_epoch : float
        The start time of the simulation in seconds.
    specific_impulse : float
        Constant specific impulse of the vehicle.  
    minimum_mars_distance : float
        Minimum distance from Mars at which the propagation stops.
    time_buffer : float
        Time interval between the simulation start epoch and the beginning of the hodographic trajectory.
    bodies : tudatpy.kernel.simulation.environment_setup.SystemOfBodies,
        System of bodies present in the simulation.
    benchmark_propagator_settings
        Propagator settings object which is used to run the benchmark propagations.
    trajectory_parameters
        List that represents the trajectory parameters for the spacecraft.
    are_dependent_variables_present : bool
        If there are dependent variables to save.
    output_path : str (default: None)
        If and where to save the benchmark results (if None, results are NOT written).

    Returns
    -------
    return_list : list
        List of state and dependent variable history in this order: state_1, state_2, dependent_1_ dependent_2.
    """
    ### CREATION OF THE TWO BENCHMARKS ###
    # Define benchmarks' step sizes
    first_benchmark_step_size = benchmark_step_size  
    second_benchmark_step_size = 2.0 * first_benchmark_step_size

    # Create integrator settings for the first benchmark, using a fixed step size RKDP8(7) integrator
    # (the minimum and maximum step sizes are set equal, while both tolerances are set to inf)
    first_benchmark_integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
        simulation_start_epoch,
        first_benchmark_step_size,
        propagation_setup.integrator.RKCoefficientSets.rkdp_87,
        first_benchmark_step_size,
        first_benchmark_step_size,
        np.inf,
        np.inf)
    # Create Low Thrust Problem object for first benchmark
    first_benchmark = LowThrust.LowThrustProblem(bodies,
                                                 first_benchmark_integrator_settings,
                                                 benchmark_propagator_settings,
                                                 specific_impulse,
                                                 minimum_mars_distance,
                                                 time_buffer,
                                                 perform_propagation=True)
    print('Running first benchmark...')
    # Set new trajectory parameters and evaluate fitness
    first_benchmark.fitness(trajectory_parameters)

    # Create integrator settings for the second benchmark in the same way
    second_benchmark_integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
        simulation_start_epoch,
        second_benchmark_step_size,
        propagation_setup.integrator.RKCoefficientSets.rkdp_87,
        second_benchmark_step_size,
        second_benchmark_step_size,
        np.inf,
        np.inf)
    # Create Low Thrust Problem object for second benchmark
    second_benchmark = LowThrust.LowThrustProblem(bodies,
                                                  second_benchmark_integrator_settings,
                                                  benchmark_propagator_settings,
                                                  specific_impulse,
                                                  minimum_mars_distance,
                                                  time_buffer,
                                                  perform_propagation=True)
    print('Running second benchmark...')
    # Set new trajectory parameters and evaluate fitness
    second_benchmark.fitness(trajectory_parameters)

    ### WRITE BENCHMARK RESULTS TO FILE ###
    # Retrieve state history
    first_benchmark_states = first_benchmark.get_last_run_propagated_state_history()
    second_benchmark_states = second_benchmark.get_last_run_propagated_state_history()
    # Write results to files
    if output_path is not None:
        save2txt(first_benchmark_states, 'benchmark_1_states.dat', output_path)
        save2txt(second_benchmark_states, 'benchmark_2_states.dat', output_path)
    # Add items to be returned
    return_list = [first_benchmark_states,
                   second_benchmark_states]

    ### DO THE SAME FOR DEPENDENT VARIABLES ###
    if are_dependent_variables_present:
        # Retrieve dependent variable history
        first_benchmark_dependent_variable = first_benchmark.get_last_run_dependent_variable_history()
        second_benchmark_dependent_variable = second_benchmark.get_last_run_dependent_variable_history()
        # Write results to file
        if output_path is not None:
            save2txt(first_benchmark_dependent_variable, 'benchmark_1_dependent_variables.dat', output_path)
            save2txt(second_benchmark_dependent_variable, 'benchmark_2_dependent_variables.dat', output_path)
        # Add items to be returned
        return_list.append(first_benchmark_dependent_variable)
        return_list.append(second_benchmark_dependent_variable)

    return return_list


def compare_benchmarks(first_benchmark: dict,
                       second_benchmark: dict,
                       output_path: str,
                       filename: str) -> dict:
    """
    It compares the results of two benchmark runs.

    It uses an 8th-order Lagrange interpolator to compare the state (or the dependent variable, depending on what is
    given as input) history. The difference is returned in form of a dictionary and, if desired, written to a file named
    filename and placed in the directory output_path.

    Parameters
    ----------
    first_benchmark : dict
        State (or dependent variable history) from the first benchmark.
    second_benchmark : dict
        State (or dependent variable history) from the second benchmark.
    output_path : str
        If and where to save the benchmark results (if None, results are NOT written).
    filename : str
        Name of the output file.

    Returns
    -------
    benchmark_difference : dict
        Interpolated difference between the two benchmarks' state (or dependent variable) history.
    """
    # Create 8th-order Lagrange interpolator for first benchmark
    benchmark_interpolator = interpolators.create_one_dimensional_interpolator(first_benchmark,
                                                                               interpolators.lagrange_interpolation(8))
    # Calculate the difference between the benchmarks
    print('Calculating benchmark differences...')
    # Initialize difference dictionaries
    benchmark_difference = dict()
    # Calculate the difference between the states and dependent variables in an iterative manner
    for second_epoch in second_benchmark.keys():
        benchmark_difference[second_epoch] = benchmark_interpolator.interpolate(second_epoch) - \
                                             second_benchmark[second_epoch]
    # Write results to files
    if output_path is not None:
        save2txt(benchmark_difference, filename, output_path)
    # Return the interpolator
    return benchmark_difference
