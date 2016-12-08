#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

parser = argparse.ArgumentParser(description='Sets up a periodic lattice of particles coupled with their nearest neighbors, shears the system and measures the shear stress.')
parser.add_argument('-N', type=int, required=False, default=4, help='Number of particles stacked vertically. Needs to be 3 or more. There is only one stack of particles, since the symmetry of the system requires the forces on the horizontal direction to cancel.')
parser.add_argument('-k', type=float, required=False, default=1, help='Prefactor k in force F=-k*d**e acting between neighboring particles with d their distance.')
parser.add_argument('-e', type=float, required=False, default=1.0, help='Exponent e in force F=-k*d**e acting between neighboring particles with d their distance. e=1 makes the bonds harmonic, e=-2 like gravity.')
parser.add_argument('-d', '--drag', type=float, required=False, default=1, help='Frictional constant coupling particles to background shear flow.')
parser.add_argument('-f', '--frequency', type=float, required=False, default=0.1, help='Frequency f of the applied sinusoidal strain gamma(t)=a*sin(2*pi*f*t).')
parser.add_argument('-g0', '--gamma0', type=float, required=False, default=1, help='Amplitude a of the oscillating strain gamma(t)=a*sin(2*pi*f*t).')
parser.add_argument('-n', type=int, required=False, default=10, help='Number of periods of the oscillatory strain.')
parser.add_argument('-dt', type=float, required=False, default=0.01, help='Time step of the symplectic euler integrator.')
args = parser.parse_args()

dt = args.dt
t_end = args.n / args.frequency

x = np.array([[0.0, 0.5+i] for i in np.arange(args.N)])
v = np.zeros([args.N, 2])

T_visualization = 500e-3

def F_stress(x, v, t):
  F_bond, stress_bond = F_stress_bond(x)
  F_drag = -args.drag * (v - u(x,t))
  stress_adv = np.einsum('ik,il', v, v)
  return F_bond + F_drag, stress_adv, stress_bond

def u(x, t):
  u = np.array([(2.0*x[:,1]/args.N-1.0) * 0.5*args.gamma0*args.N * np.cos(2.0*np.pi*args.frequency*t) * 2.0*np.pi*args.frequency, np.zeros(args.N)]).T
  return u

def F_stress_bond(x):
  d = np.roll(x, 1, axis=0) - x
  #minimum image convention
  d[:,1] -= args.N * (d[:,1] > 0.5*args.N)
  d[:,1] += args.N * (d[:,1] <= -0.5*args.N)
  #lees-edwards offset
  d[:,0] += offset * (x[:,1]+d[:,1] >= args.N)
  d[:,0] -= offset * (x[:,1]+d[:,1] < 0.0)
  F1 = args.k * d * np.sum(d**2, axis=1).reshape(len(d),1)**((args.e-1.0)/2.0)

  for i in np.arange(len(x)):
    ax[0].arrow(x[i,0], x[i,1], d[i,0], d[i,1])
  
  d = np.roll(x, -1, axis=0) - x
  #minimum image convention
  d[:,1] -= args.N * (d[:,1] > 0.5*args.N)
  d[:,1] += args.N * (d[:,1] <= -0.5*args.N)
  #lees-edwards offset
  d[:,0] += offset * (x[:,1]+d[:,1] >= args.N)
  d[:,0] -= offset * (x[:,1]+d[:,1] < 0.0)
  F2 = args.k * d * np.sum(d**2, axis=1).reshape(len(d),1)**((args.e-1.0)/2.0)

  for i in np.arange(len(x)):
    ax[0].arrow(x[i,0], x[i,1], d[i,0], d[i,1])

  stress = np.einsum('ik,il', F2, d)
  
  return F1 + F2, stress

fig, ax = plt.subplots(1, 2, sharex=False, sharey=False)
plt.ion()
plt.pause(1.0e-3)

def visualize(block=False):
  global last_visualization_time

  if block or time.clock() > last_visualization_time + T_visualization:
    ax[0].grid()
    ax[0].plot(x[:,0], x[:,1], 'bo')

    ax[0].arrow(0, 0, -offset/2, 0)
    ax[0].arrow(0, args.N, offset/2, 0)
    ax[0].plot(x[0,0]+offset, x[0,1]+args.N, 'ro')
    ax[0].plot(x[-1,0]-offset, x[-1,1]-args.N, 'ro')

    ax[0].set_ylim(-1, args.N+1)
    ax[0].set_xlim(-args.gamma0*(args.N+3)/2.0, args.gamma0*(args.N+3)/2.0)
    ax[0].set_aspect('equal', adjustable='box')

    ax[1].grid()
    ax[1].plot(result[:,0], result[:,1], '-r', label='strain')
    ax[1].set_xlim([0, t_end])

    ax[1].plot(result[:,0], result[:,2], '-b', label='adv. stress')
    ax[1].plot(result[:,0], result[:,3], '-g', label='bond stress')
    legend = ax[1].legend()
    legend.get_frame().set_alpha(0.5)
    legend.draggable()

    if block:
      plt.pause(1.0e99)
    else:
      plt.pause(1.0e-12)

    last_visualization_time = time.clock()

  ax[0].cla()
  ax[1].cla()

result = np.empty([0, 4])
last_visualization_time = 0.0

print '#time strain shear_stress_advective shear_stress_bonds'
for t in np.arange(0.0, t_end, dt):
  strain = args.gamma0 * np.sin(2.0*np.pi*args.frequency*t)
  offset = strain * args.N
  force, stress_adv, stress_bond = F_stress(x, v, t)

  print '{: <8.3e} {: <+8.3e} {: <+8.3e} {: <+8.3e}'.format(t, strain, stress_adv[0,1], stress_bond[0,1])

  result = np.vstack((result, np.array([t, strain, stress_adv[0,1], stress_bond[0,1]])))

  v += force * dt
  x += v * dt

  visualize()

with open("results/N_{N}__k_{k}__d_{drag}__f_{frequency}__g0_{gamma0}__n_{n}__dt_{dt}.pkl".format(**args.__dict__), 'w') as fp:
  pickle.dump(result, fp)

F_stress_bond(x)
visualize(block=True)
