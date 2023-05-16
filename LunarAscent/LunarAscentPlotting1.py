import numpy as np
import matplotlib.pyplot as plt


# QUESTION 1

time_steps = np.logspace(-6, 6, base=2, num=49)
time_steps = time_steps[0:-10]
errors = []
for time_step in time_steps:
    benchmark_difference = np.genfromtxt('SimulationOutput/benchmarks/benchmarks_state_difference_'+str(time_step)+'.dat',delimiter='\t').reshape([-1, 8])
    benchmark_difference_pos = benchmark_difference[:,1:4]
    benchmark_difference_pos_norm = np.linalg.norm(benchmark_difference_pos, axis=1)

    errors.append(np.max(benchmark_difference_pos_norm))

    # print(time_step, ':::', np.max(benchmark_difference_pos_norm))

# a,b = np.polyfit(time_steps, errors, 1)
# lse_fit = a*time_steps+b

# visual
plt.rcParams['font.size'] = 15
plt.figure(figsize=(10, 6))
# plot
plt.plot(time_steps, errors)
plt.scatter(time_steps, errors, color='red')
# plt.plot(time_steps, lse_fit)

# titles
plt.title('Maximum benchmark error plotted as function of different time steps')
plt.xlabel("time step ∆t (s)")
plt.ylabel("maximum benchmark error (m)")
plt.yscale('log')
plt.xscale('log')
# plt.legend()
# make
plt.tight_layout()
plt.savefig('figures/q1_max_error_per_time_step.png')
# plt.show()
plt.close()
#
#
# # visual
# plt.rcParams['font.size'] = 15
# plt.figure(figsize=(10, 6))
# # plot
# plt.plot(time_steps[0:20], errors[0:20])
#
# # titles
# plt.title('Maximum benchmark error plotted as function of different time steps')
# plt.xlabel("time (s)")
# plt.ylabel("maximum benchmark error (m)")
# plt.yscale('log')
# plt.xscale('log')
# # plt.legend()
# # make
# plt.tight_layout()
# plt.savefig('figures/q1_max_error_per_time_step_zoomed.png')
# # plt.show()
# plt.close()




time_step = 0.05
benchmark_difference = np.genfromtxt('SimulationOutput/benchmarks/benchmarks_state_difference_'+str(time_step)+'.dat', delimiter='\t').reshape([-1,8])

benchmark_time = benchmark_difference[:,0]
benchmark_difference_pos = benchmark_difference[:,1:4]
benchmark_difference_vel = benchmark_difference[:,4:7]
benchmark_difference_mass = benchmark_difference[:,7]

# visual
plt.rcParams['font.size'] = 15
plt.figure(figsize=(10, 6))
# plot
plt.plot(benchmark_time, np.linalg.norm(benchmark_difference_pos, axis=1), label='position error')
plt.plot(benchmark_time, np.linalg.norm(benchmark_difference_vel, axis=1), label='velocity error')
plt.plot(benchmark_time, benchmark_difference_mass, label='mass error')

# titles
plt.title('Benchmark state error as a function of time for ∆t = '+str(time_step)+' s')
plt.xlabel("time (s)")
plt.ylabel("Benchmark error magnitude")
plt.yscale('log')
plt.legend()
# make
plt.tight_layout()
plt.savefig('figures/q1_benchmark_difference.png')
# plt.show()
plt.close()



time_step = 0.05
benchmark_difference = np.genfromtxt('SimulationOutput/benchmarks/benchmark_1_states.dat', delimiter='\t').reshape([-1,8])

# visual
plt.rcParams['font.size'] = 12
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot(benchmark_difference[:, 1], benchmark_difference[:, 2], benchmark_difference[:, 3], label='Lambert position', linestyle='-.', color='red')
# ax.scatter(0.0, 0.0, 0.0, label="Sun", marker='o', color='yellow')

# titles
ax.set_title('Spacecraft 3D trajectory')
# ax.set_xlabel('x [10^11 m]')
# ax.set_ylabel('y [10^11 m]')
# ax.set_zlabel('z [10^11 m]')
ax.legend()

# make
plt.tight_layout()
plt.savefig('figures/q1_3d_orbit.png')
# plt.show()
plt.close()




time_step = 0.05
benchmark_states = np.genfromtxt('SimulationOutput/benchmarks/benchmark_1_states.dat', delimiter='\t').reshape([-1,8])

benchmark_time = benchmark_states[:,0]
benchmark_difference_pos = benchmark_states[:,1:4]
benchmark_difference_vel = benchmark_states[:,4:7]
benchmark_difference_mass = benchmark_states[:,7]

### POSITION
# visual
plt.rcParams['font.size'] = 15
plt.figure(figsize=(10, 6))
# plot
plt.plot(benchmark_time, benchmark_difference_pos[:,0], label='benchmark state pos (x)')
plt.plot(benchmark_time, benchmark_difference_pos[:,1], label='benchmark state pos (y)')
plt.plot(benchmark_time, benchmark_difference_pos[:,2], label='benchmark state pos (z)')

# titles
plt.title('Benchmark position in x,y and z componenent as function of time')
plt.xlabel("time (s)")
plt.ylabel("Position (m)")
# plt.yscale('log')
plt.legend()
# make
plt.tight_layout()
plt.savefig('figures/q1_pos.png')
# plt.show()
plt.close()

### VELOCITY
# visual
plt.rcParams['font.size'] = 15
plt.figure(figsize=(10, 6))
# plot
plt.plot(benchmark_time, benchmark_difference_vel[:,0], label='benchmark state vel (x)')
plt.plot(benchmark_time, benchmark_difference_vel[:,1], label='benchmark state vel (y)')
plt.plot(benchmark_time, benchmark_difference_vel[:,2], label='benchmark state vel (z)')

# titles
plt.title('Benchmark velocity in x,y and z componenent as function of time')
plt.xlabel("time (s)")
plt.ylabel("Position (m)")
# plt.yscale('log')
plt.legend()
# make
plt.tight_layout()
plt.savefig('figures/q1_vel.png')
# plt.show()
plt.close()

### mass
# visual
plt.rcParams['font.size'] = 15
plt.figure(figsize=(10, 6))
# plot
plt.plot(benchmark_time, benchmark_difference_mass, label='benchmark state vel (x)')

# titles
plt.title('Benchmark mass of the vehicle as function of time')
plt.xlabel("time (s)")
plt.ylabel("mass (kg)")
# plt.yscale('log')
plt.legend()
# make
plt.tight_layout()
plt.savefig('figures/q1_mass.png')
# plt.show()
plt.close()








