"""
Python code for the floating wind farm installation design problem.

Copyright (c) 2022. Harold Van Heukelum
"""

import pathlib
from math import ceil
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import pi
from scipy.interpolate import pchip_interpolate
from scipy.optimize import fsolve

from genetic_algorithm_pfm import GeneticAlgorithm

w1 = 0.5   #City of Antwerp
w2 = 0.5   #Lantis(Project manager)
w3 = 0.0   #Inhabitants
w4 = 0.0   #Enviromental Group
w5 = 0.25   #Contractor(Not used currently)

# todo: change the points and preference scores according to the case at hand
# The Preference scores (p_points) and corresponding Objective results (x_points)
X_POINTS_COST, P_POINTS_COST = [[150, 575, 1000], [100, 60, 0]]         #Cost (M€)
X_POINTS_CAPACITY, P_POINTS_CAPACITY = [[2, 4, 10], [0, 50, 100]]       #Capacity (k)
X_POINTS_3, P_POINTS_3 = [[20, 30, 40], [0, 70, 100]]                    #Distance (km)
X_POINTS_4, P_POINTS_4 = [[6, 8, 10], [0, 20, 100]]

# todo: change the bounds according to the case at hand
# set bounds for all variables
b1 = [1500, 2000]        #Tunnel Length(m) X1
b2 = [2, 10]             #Lanes X2
b3 = [20, 40]       #Distance(km) X3
b4 = [6, 10]        #Lanes X4
bounds = [b1, b2, b3, b4]

# todo: change the variable names according to the case at hand
strCost = 'Cost'
strCapacity = 'Capacity'
str3 = 'Distance'
str4 = 'Flight movements'
strTitleXCost = strCost + ' (M€)'
strTitleXCapacity = strCapacity + ' (k)'
strTitleX3 = str3 + ' (km)'
strTitleX4 = str4 + ' (x100k)'
strTitleY = 'Preference score'

def objective_p1(x1, x2, x3, x4):
    """
    Objective to minimize the cost.

    :param x1: 1st design variable
    :param x2: 2nd design variable
    """
    return pchip_interpolate(X_POINTS_COST, P_POINTS_COST, (x1 * x2 * 0.05))


def objective_p2(x1, x2, x3, x4):
    """
    Objective to maximize the capacity.

    :param x1: 1st design variable
    :param x2: 2nd design variable
    """
    return pchip_interpolate(X_POINTS_CAPACITY, P_POINTS_CAPACITY, (x2))


def objective_p3(x1, x2, x3, x4):
    """
    Objective to maximize the shopping potential preference.

    :param x1: 1st design variable
    :param x2: 2nd design variable
    """
    return pchip_interpolate(X_POINTS_3, P_POINTS_3, (x3))

def objective_p4(x1, x2, x3, x4):
    """
    Objective to maximize the shopping potential preference.

    :param x1: 1st design variable
    :param x2: 2nd design variable
    """
    return pchip_interpolate(X_POINTS_4, P_POINTS_4, (x4))

def objective(variables):
    """
    Objective function that is fed to the GA. Calles the separate preference functions that are declared above.

    :param variables: array with design variable values per member of the population. Can be split by using array
    slicing
    :return: 1D-array with aggregated preference scores for the members of the population.
    """
    # extract 1D design variable arrays from full 'variables' array
    x1 = variables[:, 0]
    x2 = variables[:, 1]
    x3 = variables[:, 2]
    x4 = variables[:, 3]

    # calculate the preference scores
    p_1 = objective_p1(x1, x2, x3, x4)
    p_2 = objective_p2(x1, x2, x3, x4)
    p_3 = objective_p3(x1, x2, x3, x4)
    p_4 = objective_p4(x1, x2, x3, x4)

    # aggregate preference scores and return this to the GA
    return [w1, w2, w3, w4], [p_1, p_2, p_3, p_4]

# todo: change the constraints according to the case at hand
def constraint_1(variables):
    """Constraint that checks if the sum of the areas x1 and x2 is not higher than 10,000 m2.

    :param variables: ndarray of n-by-m, with n the population size of the GA and m the number of variables.
    :return: list with scores of the constraint
    """
    x1 = variables[:, 0]
    x2 = variables[:, 1]
    x3 = variables[:, 2]
    x4 = variables[:, 3]
    # Capacity should be no more than 6 times the cost
    return x2 * 6 - x1  # < 0

def constraint_2(variables):
    """Constraint that checks if the sum of the areas x1 and x2 is not lower than 3,000 m2.

    :param variables: ndarray of n-by-m, with n the population size of the GA and m the number of variables.
    :return: list with scores of the constraint
    """
    x1 = variables[:, 0]
    x2 = variables[:, 1]
    x3 = variables[:, 2]
    x4 = variables[:, 3]

    return 15 + 0.15*(x4 - 6) + 0.2*(x3-10) - x1  # < 0


# define list with constraints
#cons = [['ineq', constraint_1], ['ineq', constraint_2]]
cons = []

# create arrays for plotting continuous preference curves
c1 = np.linspace(X_POINTS_COST[0], X_POINTS_COST[-1])
c2 = np.linspace(X_POINTS_CAPACITY[0], X_POINTS_CAPACITY[-1])
c3 = np.linspace(X_POINTS_3[0], X_POINTS_3[-1])
c4 = np.linspace(X_POINTS_4[0], X_POINTS_4[-1])

# calculate the preference functions
p1 = pchip_interpolate(X_POINTS_COST, P_POINTS_COST, (c1))
p2 = pchip_interpolate(X_POINTS_CAPACITY, P_POINTS_CAPACITY, (c2))
p3 = pchip_interpolate(X_POINTS_3, P_POINTS_3, (c3))
p4 = pchip_interpolate(X_POINTS_4, P_POINTS_4, (c4))

# create figure that plots all preference curves and the preference scores of the returned results of the GA
fig = plt.figure(figsize=((10,10)))

font1 = {'size':20}
font2 = {'size':15}

plt.rcParams['font.size'] = '12'
plt.rcParams['savefig.dpi'] = 300

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(c1, p1, label='Preference curve', color='black')
ax1.set_xlim((X_POINTS_COST[0], X_POINTS_COST[-1]))
ax1.set_ylim((0, 102))
ax1.set_title('City of Antwerp')
ax1.set_xlabel(strTitleXCost)
ax1.set_ylabel(strTitleY)
ax1.grid()
ax1.legend()
ax1.grid(linestyle = '--')

#fig = plt.figure()
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(c2, p2, label='Preference curve', color='black')
ax2.set_xlim((X_POINTS_CAPACITY[0], X_POINTS_CAPACITY[-1]))
ax2.set_ylim((0, 102))
ax2.set_title('Inhabitants')
ax2.set_xlabel(strTitleXCapacity)
ax2.set_ylabel(strTitleY)
ax2.grid()
ax2.legend()
ax2.grid(linestyle = '--')

#fig = plt.figure()
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(c3, p3, label='Preference curve', color='black')
ax3.set_xlim((X_POINTS_3[0], X_POINTS_3[-1]))
ax3.set_ylim((0, 102))
ax3.set_title('Ministry of Environment')
ax3.set_xlabel(strTitleX3)
ax3.set_ylabel(strTitleY)
ax3.grid()
ax3.legend()
ax3.grid(linestyle = '--')

#fig = plt.figure()
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(c4, p4, label='Preference curve', color='black')
ax4.set_xlim((X_POINTS_4[0], X_POINTS_4[-1]))
ax4.set_ylim((0, 102))
ax4.set_title('Airport')
ax4.set_xlabel(strTitleX4)
ax4.set_ylabel(strTitleY)
ax4.grid()
ax4.legend()
ax4.grid(linestyle = '--')

ax1.legend()
ax2.legend()
fig.tight_layout()

#Two  lines to make our compiler able to draw:
fig.savefig("Oosterweel.png")

# We run the optimization with two paradigms
paradigm = ['minmax', 'tetra']
marker = ['o', '*']
colours = ['orange', 'green']

# Define the figure and axes before the loop
fig = plt.figure(figsize=(12, 8))

# Creating four subplots for the four preference scores
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

 # Already defined above
# # Create arrays for plotting continuous preference curves
# c1 = np.linspace(15, 40)
# c2 = np.linspace(0.5, 0.9)
# c3 = np.linspace(20, 40)
# c4 = np.linspace(6, 10)

# # Calculate the preference functions
# p1 = pchip_interpolate([15, 20, 40], [100, 20, 0], c1)
# p2 = pchip_interpolate([0.5, 0.7, 0.9], [100, 45, 0], c2)
# p3 = pchip_interpolate([20, 30, 40], [0, 70, 100], c3)
# p4 = pchip_interpolate([6, 8, 10], [0, 20, 100], c4)

# Plot each preference curve on the respective subplot
ax1.plot(c1, p1, label='Preference curve', color='black')
ax2.plot(c2, p2, label='Preference curve', color='black')
ax3.plot(c3, p3, label='Preference curve', color='black')
ax4.plot(c4, p4, label='Preference curve', color='black')

for i in range(2):
    # Dictionary with parameter settings for the GA run with the IMAP solver
    options = {
        'n_bits': 8,
        'n_iter': 400,
        'n_pop': 500,
        'r_cross': 0.8,
        'max_stall': 8,
        'aggregation': paradigm[i],  # minmax or a_fine
        "var_type_mixed": ["int", "int", "real", "real"],
    }

    # Run the GA and print its result
    print(f'Run GA with {paradigm[i]}')
    ga = GeneticAlgorithm(objective=objective, constraints=cons, bounds=bounds, options=options)
    score_IMAP, design_variables_IMAP, _ = ga.run()

    # Print the optimal result in a readable format
    print(f'Optimal result for x1 = {round(design_variables_IMAP[0], 2)} meters in length and '
          f'x2 = {round(design_variables_IMAP[1], 2)} lanes and '
          f'x3 = {round(design_variables_IMAP[2], 2)} kilometers and '
          f'x4 = {round(design_variables_IMAP[3], 2)} flight movements')

    # todo: calculate the individual preference scores for the results
    # Calculate individual preference scores for the results
    c1_res = design_variables_IMAP[0] * design_variables_IMAP[1] * 0.05
    p1_res = pchip_interpolate(X_POINTS_COST, P_POINTS_COST, c1_res)

    c2_res = design_variables_IMAP[1]
    p2_res = pchip_interpolate(X_POINTS_CAPACITY, P_POINTS_CAPACITY, c2_res)

    c3_res = design_variables_IMAP[2]
    p3_res = pchip_interpolate(X_POINTS_3, P_POINTS_3, c3_res)

    c4_res = design_variables_IMAP[3]
    p4_res = pchip_interpolate(X_POINTS_4, P_POINTS_4, c4_res)

    # Debugging prints to check calculated values
    print(f"c1_res: {c1_res}, p1_res: {p1_res}")
    print(f"c2_res: {c2_res}, p2_res: {p2_res}")
    print(f"c3_res: {c3_res}, p3_res: {p3_res}")
    print(f"c4_res: {c4_res}, p4_res: {p4_res}")

    # Plot the results on the preference curve subplots
    ax1.scatter(c1_res, p1_res, label='Optimal solution ' + paradigm[i], color=colours[i], marker=marker[i])
    ax2.scatter(c2_res, p2_res, label='Optimal solution ' + paradigm[i], color=colours[i], marker=marker[i])
    ax3.scatter(c3_res, p3_res, label='Optimal solution ' + paradigm[i], color=colours[i], marker=marker[i])
    ax4.scatter(c4_res, p4_res, label='Optimal solution ' + paradigm[i], color=colours[i], marker=marker[i])

# Add legends and set titles for each subplot
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

ax1.set_title('Optimal Solution for ' + strCost + ' (x1)')
ax1.set_xlabel(strTitleXCost)
ax1.set_ylabel(strTitleY)

ax2.set_title('Optimal Solution for ' + strCapacity + ' (x2)')
ax2.set_xlabel(strTitleXCapacity)
ax2.set_ylabel(strTitleY)

ax3.set_title('Optimal Solution for ' + str3 + ' (x3)')
ax3.set_xlabel(strTitleX3)
ax3.set_ylabel(strTitleY)

ax4.set_title('Optimal Solution for ' + str4 + ' (x4)')
ax4.set_xlabel(strTitleX4)
ax4.set_ylabel(strTitleY)

# Adjust the layout
fig.tight_layout()

# Display the plot
plt.show()