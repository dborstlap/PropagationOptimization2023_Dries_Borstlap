import numpy as np
import matplotlib.pyplot as plt
import LunarAscentUtilities as Util


threshold = 0.1

n_propagators = 7
n_step_size_settings = 5
integrator_index = 1


all_results = {}
# results_n_evaluations = {}
# ancillary_simulation_info = {}


propagator_names = [
    'cowell',
    'encke',
    'gauss_keplerian',
    'gauss_modified_equinoctial',
    'unified_state_model_quaternions',
    'unified_state_model_modified_rodrigues_parameters',
    'unified_state_model_exponential_map',
]



for propagator_index in range(n_propagators): #
    # visual
    plt.rcParams['font.size'] = 15
    plt.figure(figsize=(10, 6))
    for step_size_index in range(n_step_size_settings):
        step_size_name = [
            'step size ' + str(format(2 ** (-7 + step_size_index), ".0e")),
            'step size ' + str(format(2 ** (-1 + step_size_index), ".0e")),
            'step size ' + str(format(2 ** (0 + step_size_index), ".0e")),
            'tolerance ' + str(format(10.0 ** (-12.0 + step_size_index), ".0e")),
            'tolerance ' + str(format(10.0 ** (-12.0 + step_size_index), ".0e")),
            'tolerance ' + str(format(10.0 ** (-16.0 + step_size_index), ".0e")),
            'tolerance ' + str(format(10 ** (-8 + step_size_index), ".0e")),
            'tolerance ' + str(format(10 ** (-6 + step_size_index), ".0e")),
            'tolerance ' + str(format(10 ** (-6 + step_size_index), ".0e")),
            'step size ' + str(format(2 ** (2 + step_size_index), ".0e")),
            'step size ' + str(format(2 ** (2 + step_size_index), ".0e")),
            'step size ' + str(format(2 ** (-1 + step_size_index), ".0e")),
            'tolerance ' + str(format(10 ** (-13 + step_size_index), ".0e")),
            'tolerance ' + str(format(10 ** (-13 + step_size_index), ".0e")),
            'tolerance ' + str(format(10 ** (-12 + step_size_index), ".0e")),
        ]


        # print progress
        print('prop:'+str(propagator_index)+' int:'+str(integrator_index)+' step_size:'+str(step_size_index))

        # file paths
        path = 'SimulationOutput/prop_'+str(propagator_index)+'/int_'+str(integrator_index)+'/step_size_'+str(step_size_index)+'/'
        file_ancillary_simulation_info = path + 'ancillary_simulation_info.txt'
        file_dependent_variable_difference_wrt_benchmark = path + 'dependent_variable_difference_wrt_benchmark.dat'
        file_dependent_variable_history = path + 'dependent_variable_history.dat'
        file_state_difference_wrt_benchmark = path + 'state_difference_wrt_benchmark.dat'
        file_state_history = path + 'state_history.dat'
        file_unprocessed_state_history = path + 'unprocessed_state_history.dat'

        # convert data to dictionaries
        ancillary_simulation_info                   = Util.read_ancillary_info(file_ancillary_simulation_info)
        dependent_variable_difference_wrt_benchmark = Util.read_data_file(file_dependent_variable_difference_wrt_benchmark, 3)
        dependent_variable_history                  = Util.read_data_file(file_dependent_variable_history, 3)
        state_difference_wrt_benchmark              = Util.read_data_file(file_state_difference_wrt_benchmark, 7)
        state_history                               = Util.read_data_file(file_state_history, 7)
        # unprocessed_state_history                   = Util.read_data_file(file_unprocessed_state_history, 7)

        # store in results for this run
        results = {}
        results['ancillary_simulation_info'] = ancillary_simulation_info
        results['dependent_variable_difference_wrt_benchmark'] = dependent_variable_difference_wrt_benchmark
        results['dependent_variable_history'] = dependent_variable_history
        results['state_difference_wrt_benchmark'] = state_difference_wrt_benchmark
        results['state_history'] = state_history
        # results['unprocessed_state_history'] = unprocessed_state_history

        name = 'prop_' + str(propagator_index) + '_int_' + str(integrator_index) + '_step_size_' + str(step_size_index)
        all_results[name] = results


        step_name = step_size_name[integrator_index]
        n_func_evals = str(int(int(ancillary_simulation_info['Number of function evaluations'])))
        label = step_name + ' (' + n_func_evals + ')'


        # plot
        time = state_difference_wrt_benchmark[:,0]
        pos = state_difference_wrt_benchmark[:, 1:4]
        pos_norm = np.linalg.norm(pos, axis=1)
        plt.plot(time, pos_norm, label=label)
        # plt.scatter(time, pos_norm)

    # plot guides
    end_time = state_history[:,0][-1]
    plt.plot([0,400], [threshold, threshold], label='maximum threshold')
    plt.plot([0, 400], [threshold/100*2, threshold/100*2], label='benchmark dominant threshold')

    # titles
    plt.title(propagator_names[integrator_index])
    plt.xlabel("time (s)")
    plt.ylabel("Position error (m)")
    plt.yscale('log')
    plt.legend()
    # make
    plt.tight_layout()
    plt.savefig('figures/'+'q3_'+str(propagator_index)+'.png')
    # plt.show()
    plt.close()