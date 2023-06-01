import numpy as np
import matplotlib.pyplot as plt


# # for model 1
# depend_var = np.genfromtxt('Model_1/dependent_variable_history.dat', delimiter='\t').reshape([-1, 7])
# time = depend_var[:,0]
# acc_moon = depend_var[:,1]
# acc_earth = depend_var[:,2]
# acc_venus = depend_var[:,3]
# acc_mars = depend_var[:,4]
# acc_jupiter = depend_var[:,5]
# acc_sun = depend_var[:,6]
# # acc_cannon = depend_var[:,7]
# # acc_thrust = depend_var[:,8]
#
#
# plt.rcParams['font.size'] = 15
# plt.figure(figsize=(10, 6))
#
# plt.plot(time, acc_moon, label='moon')
# plt.plot(time, acc_earth, label='earth')
# plt.plot(time, acc_venus, label='venus')
# plt.plot(time, acc_mars, label='mars')
# plt.plot(time, acc_jupiter, label='jupiter')
# plt.plot(time, acc_sun, label='sun')
# # plt.plot(time, acc_cannon, label='cannon')
# # plt.plot(time, acc_thrust, label='thrust')
#
# plt.title('Magnitude of accelerations wrt to moon, caused by different bodies')
# plt.xlabel("time (s)")
# plt.ylabel("Acceleration (m/s^2)")
# plt.yscale('log')
# plt.legend()
#
# plt.tight_layout()
# plt.show()


numer_of_models = 16
for model_number in range(1, numer_of_models):
    state_diff = np.genfromtxt('Model_' + str(model_number) + '/state_difference_wrt_nominal_case.dat', delimiter='\t').reshape([-1, 8])
    pos_dif = np.linalg.norm(state_diff[:,1:4], axis=1)
    print('state_diff for model '+str(model_number)+' = ', pos_dif[-209])




labels = [
    'Moon point mass',
    'Moon spherical harmonics [1,1]',
    'Moon spherical harmonics [2,2]',
    'Moon spherical harmonics [10,10]',
    'Moon spherical harmonics [100,100]',
    'earth point mass',
    'Earth spherical harmonics [1,1]',
    'Earth spherical harmonics [2,2]',
    'Earth spherical harmonics [10,10]',
    'Earth spherical harmonics [100,100]',
    'Sun point mass gravity',
    'Venus point mass gravity',
    'Mars point mass gravity',
    'Jupiter point mass gravity',
    'Sun, venus, mars, Jupiter and earth point mass gravity',
]

# visual
plt.rcParams['font.size'] = 15
plt.figure(figsize=(10, 6))
for model_number in range(10,14):   #  (1, 5)   (5,10)   (10,14)
    state_diff = np.genfromtxt('Model_' + str(model_number) + '/state_difference_wrt_nominal_case.dat', delimiter='\t').reshape([-1, 8])
    state_diff = state_diff[:-20]
    pos_dif = np.linalg.norm(state_diff[:,1:4], axis=1)
    time = state_diff[:,0]
    plt.plot(time, pos_dif, label=labels[model_number])

# titles
plt.title('Position difference as a function of time')
plt.xlabel("time (s)")
plt.ylabel("Position difference (m)")
# plt.yscale('log')
plt.ylim(0,0.008)
plt.legend()

# make
plt.tight_layout()
plt.savefig('figures/q1_planets.png')
# plt.show()
plt.close()

























