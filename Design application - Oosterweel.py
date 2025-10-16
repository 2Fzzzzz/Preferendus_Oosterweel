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

w1 = 0.2   #City of Antwerp
w2 = 0.2   #Inhabitants
w3 = 0.2   #Lantis(Project manager)
w4 = 0.2   #Enviromental Group
w5 = 0.2   #Contractor

# todo: change the points and preference scores according to the case at hand
# The Preference scores (p_points) and corresponding Objective results (x_points)
X_POINTS_COST, P_POINTS_COST = [[300, 325, 600], [100, 60, 0]]         #Cost (M€)
X_POINTS_CAPACITY, P_POINTS_CAPACITY = [[4000, 11000, 18000], [0, 50, 100]]       #Capacity
X_POINTS_ConsTime, P_POINTS_ConsTime = [[6.8, 12, 27.1], [100, 40, 0]]     #Construction time
X_POINTS_CO2, P_POINTS_CO2 = [[93.84, 173.0, 252], [100, 30, 0]]      #CO2 emissions (kton)   
X_POINTS_Profit, P_POINTS_Profit = [[60, 90, 120], [0, 40, 100]]      #Profit (M€)

# todo: change the bounds according to the case at hand
# set bounds for all variables
b1 = [1500, 2000]       #Tunnel Length(m) X1
b2 = [4, 6]             #Lanes X2
b3 = [5.8, 6.0]         #Inner Height(m) X3
b4 = [1.2, 1.5]         #Thickness(m) X4
b5 = [2.8, 3.7]         #Lane width(m) X5
b6 = [50, 100]          #Speed limit(km/h) X6
b7 = [10, 30]          #Density(cars/km) X7
b8 = [1, 10]            #Number of machines X8
b9 = [0.2, 1.0]         #Politian factor X9
bounds = [b1, b2, b3, b4, b5, b6, b7, b8, b9]

# Actual number of variables
b1Len = 1800
b2Lanes = 6
b3Height = 5.8 
b4Thickness = 1.5 # we dont know
b5LaneWidth = 3.5 # we dont know
b6SpeedLimit = 80 # we dont know
b7Density = 20.0 # we dont know
b8Machines = 8 # we dont know
b9Politian = 1.0 # we dont know
TotalHeight = 10.0 # = InnerHeight + 2*Thickness + 1m ballast
TotalWidth = 42.0
ConstructionTime = 7.0
TotalCost = 500  # we dont know
TotalCO2 = 210 # kton
Capacity = 9300 # we dont know

start_Points_population = [b1Len, b2Lanes, b3Height, b4Thickness, b5LaneWidth, b6SpeedLimit, b7Density, b8Machines, b9Politian]
#start_Points_population = [325, 60000, 12, 136.00, 90]

# todo: change the variable names according to the case at hand
strCost = 'Cost'
strCapacity = 'Capacity'
str3 = 'Construction time'
str4 = 'CO2 emissions'
str5 = 'Profit'
strTitleXCost = strCost + ' (M€)'
strTitleXCapacity = strCapacity + ' (cars/h)'
strTitleX3 = str3 + ' (years)'
strTitleX4 = str4 + ' (kton)'
strTitleX5 = str5 + ' (M€)'
strTitleY = 'Preference score'

def calculate_cost(x1, x2, x3, x4, x5, x6, x7, x8, x9):
    return x1 * x2 * 0.05  # in M€
def calculate_capacity(x1, x2, x3, x4, x5, x6, x7, x8, x9):
    return x2 * x7 * x6
def calculate_construction_time(x1, x2, x3, x4, x5, x6, x7, x8, x9):
    return (x1 * x2 / (x8 * x9)) ** 0.3
def calculate_CO2_emissions(x1, x2, x3, x4, x5, x6, x7, x8, x9):
    return x1 * (x3 + 2 * x4 + 1) * (x2 * x5 + 2 * x4) * 0.0005 # in kton
def calculate_profit(x1, x2, x3, x4, x5, x6, x7, x8, x9):
    return calculate_cost(x1, x2, x3, x4, x5, x6, x7, x8, x9) * 0.2

CalcCost = calculate_cost(b1Len, b2Lanes, b3Height, b4Thickness, b5LaneWidth, b6SpeedLimit, b7Density, b8Machines, b9Politian)
CalcCapacity = calculate_capacity(b1Len, b2Lanes, b3Height, b4Thickness, b5LaneWidth, b6SpeedLimit, b7Density, b8Machines, b9Politian)
CalcConstructionTime = calculate_construction_time(b1Len, b2Lanes, b3Height, b4Thickness, b5LaneWidth, b6SpeedLimit, b7Density, b8Machines, b9Politian)
CalcCO2 = calculate_CO2_emissions(b1Len, b2Lanes, b3Height, b4Thickness, b5LaneWidth, b6SpeedLimit, b7Density, b8Machines, b9Politian)
CalcProfit = calculate_profit(b1Len, b2Lanes, b3Height, b4Thickness, b5LaneWidth, b6SpeedLimit, b7Density, b8Machines, b9Politian)

def objective_p1(x1, x2, x3, x4, x5, x6, x7, x8, x9):
    """
    Objective to minimize the cost.

    :param x1: 1st design variable
    :param x2: 2nd design variable
    """
    return pchip_interpolate(X_POINTS_COST, P_POINTS_COST, (calculate_cost(x1, x2, x3, x4, x5, x6, x7, x8, x9)))


def objective_p2(x1, x2, x3, x4, x5, x6, x7, x8, x9):
    """
    Objective to maximize the capacity.

    :param x1: 1st design variable
    :param x2: 2nd design variable
    """
    return pchip_interpolate(X_POINTS_CAPACITY, P_POINTS_CAPACITY, (calculate_capacity(x1, x2, x3, x4, x5, x6, x7, x8, x9)))


def objective_p3(x1, x2, x3, x4, x5, x6, x7, x8, x9):
    """
    Objective to maximize the shopping potential preference.

    :param x1: 1st design variable
    :param x2: 2nd design variable
    """
    return pchip_interpolate(X_POINTS_ConsTime, P_POINTS_ConsTime, (calculate_construction_time(x1, x2, x3, x4, x5, x6, x7, x8, x9)))

def objective_p4(x1, x2, x3, x4, x5, x6, x7, x8, x9):
    """
    Objective to maximize the shopping potential preference.

    :param x1: 1st design variable
    :param x2: 2nd design variable
    """
    return pchip_interpolate(X_POINTS_CO2, P_POINTS_CO2, (calculate_CO2_emissions(x1, x2, x3, x4, x5, x6, x7, x8, x9)))

def objective_p5(x1, x2, x3, x4, x5, x6, x7, x8, x9):
    """
    Objective to maximize the shopping potential preference.

    :param x1: 1st design variable
    :param x2: 2nd design variable
    """
    return pchip_interpolate(X_POINTS_Profit, P_POINTS_Profit, (calculate_profit(x1, x2, x3, x4, x5, x6, x7, x8, x9)))

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
    x5 = variables[:, 4]
    x6 = variables[:, 5]
    x7 = variables[:, 6]
    x8 = variables[:, 7]
    x9 = variables[:, 8]

    # calculate the preference scores
    p_1 = objective_p1(x1, x2, x3, x4, x5, x6, x7, x8, x9)
    p_2 = objective_p2(x1, x2, x3, x4, x5, x6, x7, x8, x9)
    p_3 = objective_p3(x1, x2, x3, x4, x5, x6, x7, x8, x9)
    p_4 = objective_p4(x1, x2, x3, x4, x5, x6, x7, x8, x9)
    p_5 = objective_p5(x1, x2, x3, x4, x5, x6, x7, x8, x9)

    # aggregate preference scores and return this to the GA
    return [w1, w2, w3, w4, w5], [p_1, p_2, p_3, p_4, p_5]

# todo: change the constraints according to the case at hand
def constraint_Speed(variables):
    """
    :param variables: ndarray of n-by-m, with n the population size of the GA and m the number of variables.
    :return: list with scores of the constraint
    """
    x1 = variables[:, 0]
    x2 = variables[:, 1]
    x3 = variables[:, 2]
    x4 = variables[:, 3]
    x5 = variables[:, 4]
    x6 = variables[:, 5]
    x7 = variables[:, 6]
    x8 = variables[:, 7]
    x9 = variables[:, 8]
    # Speed limit should be at least 13 times the number of lanes
    return x6 - x2 * 13 # < 0

def constraint_HeightWidth(variables):
    """
    :param variables: ndarray of n-by-m, with n the population size of the GA and m the number of variables.
    :return: list with scores of the constraint
    """
    x1 = variables[:, 0]
    x2 = variables[:, 1]
    x3 = variables[:, 2]
    x4 = variables[:, 3]
    x5 = variables[:, 4]
    x6 = variables[:, 5]
    x7 = variables[:, 6]
    x8 = variables[:, 7]
    x9 = variables[:, 8]
    # Height should be at least 0.04 times the total width
    return 0.04 * (x2 * x5) - x4 # < 0

def constraint_WidthSpeed(variables):
    """
    :param variables: ndarray of n-by-m, with n the population size of the GA and m the number of variables.
    :return: list with scores of the constraint
    """
    x1 = variables[:, 0]
    x2 = variables[:, 1]
    x3 = variables[:, 2]
    x4 = variables[:, 3]
    x5 = variables[:, 4]
    x6 = variables[:, 5]
    x7 = variables[:, 6]
    x8 = variables[:, 7]
    x9 = variables[:, 8]
    # Total width should be less than speed limit
    return x2 * x5 - x6 # < 0

def constraint_WidthThicknessHeight(variables):
    """
    :param variables: ndarray of n-by-m, with n the population size of the GA and m the number of variables.
    :return: list with scores of the constraint
    """
    x1 = variables[:, 0]
    x2 = variables[:, 1]
    x3 = variables[:, 2]
    x4 = variables[:, 3]
    x5 = variables[:, 4]
    x6 = variables[:, 5]
    x7 = variables[:, 6]
    x8 = variables[:, 7]
    x9 = variables[:, 8]
    # Lane Width plus Thickness should be less than Height
    return x4 + x5 - x3 # < 0

def constraint_DensitySpeed(variables):
    """
    :param variables: ndarray of n-by-m, with n the population size of the GA and m the number of variables.
    :return: list with scores of the constraint
    """
    x1 = variables[:, 0]
    x2 = variables[:, 1]
    x3 = variables[:, 2]
    x4 = variables[:, 3]
    x5 = variables[:, 4]
    x6 = variables[:, 5]
    x7 = variables[:, 6]
    x8 = variables[:, 7]
    x9 = variables[:, 8]
    # Politian factor should be less than 0.1 times the number of lanes
    return 2000 / x7 - x6 # < 0

def constraint_LanesDensity(variables):
    """
    :param variables: ndarray of n-by-m, with n the population size of the GA and m the number of variables.
    :return: list with scores of the constraint
    """
    x1 = variables[:, 0]
    x2 = variables[:, 1]
    x3 = variables[:, 2]
    x4 = variables[:, 3]
    x5 = variables[:, 4]
    x6 = variables[:, 5]
    x7 = variables[:, 6]
    x8 = variables[:, 7]
    x9 = variables[:, 8]
    # Number of Density should be less than 10000 / (Lanes^3)
    return 10000 / x2**3 - x7 # < 0

# todo: define list with constraints
#cons = []
cons = [['ineq', constraint_Speed], ['ineq', constraint_HeightWidth], ['ineq', constraint_WidthSpeed], ['ineq', constraint_WidthThicknessHeight]
        , ['ineq', constraint_DensitySpeed], ['ineq', constraint_LanesDensity]]

# create arrays for plotting continuous preference curves
c1 = np.linspace(X_POINTS_COST[0], X_POINTS_COST[-1])
c2 = np.linspace(X_POINTS_CAPACITY[0], X_POINTS_CAPACITY[-1])
c3 = np.linspace(X_POINTS_ConsTime[0], X_POINTS_ConsTime[-1])
c4 = np.linspace(X_POINTS_CO2[0], X_POINTS_CO2[-1])
c5 = np.linspace(X_POINTS_Profit[0], X_POINTS_Profit[-1])

# calculate the preference functions
p1 = pchip_interpolate(X_POINTS_COST, P_POINTS_COST, (c1))
p2 = pchip_interpolate(X_POINTS_CAPACITY, P_POINTS_CAPACITY, (c2))
p3 = pchip_interpolate(X_POINTS_ConsTime, P_POINTS_ConsTime, (c3))
p4 = pchip_interpolate(X_POINTS_CO2, P_POINTS_CO2, (c4))
p5 = pchip_interpolate(X_POINTS_Profit, P_POINTS_Profit, (c5))

# create figure that plots all preference curves and the preference scores of the returned results of the GA
fig = plt.figure(figsize=((10,10)))

font1 = {'size':20}
font2 = {'size':15}

plt.rcParams['font.size'] = '12'
plt.rcParams['savefig.dpi'] = 300

ax1 = fig.add_subplot(2, 3, 1)
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
ax2 = fig.add_subplot(2, 3, 2)
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
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(c3, p3, label='Preference curve', color='black')
ax3.set_xlim((X_POINTS_ConsTime[0], X_POINTS_ConsTime[-1]))
ax3.set_ylim((0, 102))
ax3.set_title('Lantis')
ax3.set_xlabel(strTitleX3)
ax3.set_ylabel(strTitleY)
ax3.grid()
ax3.legend()
ax3.grid(linestyle = '--')

#fig = plt.figure()
ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(c4, p4, label='Preference curve', color='black')
ax4.set_xlim((X_POINTS_CO2[0], X_POINTS_CO2[-1]))
ax4.set_ylim((0, 102))
ax4.set_title('Enviromental Group')
ax4.set_xlabel(strTitleX4)
ax4.set_ylabel(strTitleY)
ax4.grid()
ax4.legend()
ax4.grid(linestyle = '--')

#fig = plt.figure()
ax5 = fig.add_subplot(2, 3, 5)
ax5.plot(c5, p5, label='Preference curve', color='black')
ax5.set_xlim((X_POINTS_Profit[0], X_POINTS_Profit[-1]))
ax5.set_ylim((0, 102))
ax5.set_title('Contractors')
ax5.set_xlabel(strTitleX5)
ax5.set_ylabel(strTitleY)
ax5.grid()
ax5.legend()
ax5.grid(linestyle = '--')

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()
fig.tight_layout()

#Two  lines to make our compiler able to draw:
fig.savefig("Oosterweel.png")

# We run the optimization with two paradigms
paradigm = ['minmax', 'tetra']
marker = ['o', '*', 's', '^']
colours = ['orange', 'green', 'red', 'blue']

# Define the figure and axes before the loop
fig = plt.figure(figsize=(12, 8))

# Creating four subplots for the four preference scores
ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)

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
ax5.plot(c5, p5, label='Preference curve', color='black')

for i in range(2):
    # Dictionary with parameter settings for the GA run with the IMAP solver
    options = {
        'n_bits': 8,
        'n_iter': 400,
        'n_pop': 500,
        'r_cross': 0.8,
        'max_stall': 8,
        'aggregation': paradigm[i],  # minmax or a_fine
        "var_type_mixed": ["int", "int", "real", "real", "real", "int", "real", "int", "real"],
    }

    # Run the GA and print its result
    print(f'Run GA with {paradigm[i]}')
    ga = GeneticAlgorithm(objective=objective, constraints=cons, bounds=bounds, options=options, start_points_population=
                          [start_Points_population])
    score_IMAP, design_variables_IMAP, _ = ga.run()

    # Print the optimal result in a readable format
    print(f'Optimal method {paradigm[i]}, result for x1 = {round(design_variables_IMAP[0], 2)} meters in length and '
          f'x2 = {round(design_variables_IMAP[1], 2)} lanes and '
          f'x3 = {round(design_variables_IMAP[2], 2)} meters in height and '
          f'x4 = {round(design_variables_IMAP[3], 2)} meters in thickness and '
          f'x5 = {round(design_variables_IMAP[4], 2)} meters in lane width and '
          f'x6 = {round(design_variables_IMAP[5], 2)} km/h in speed limit and '
          f'x7 = {round(design_variables_IMAP[6], 2)} cars/km in density and '
          f'x8 = {round(design_variables_IMAP[7], 2)} machines and '
          f'x9 = {round(design_variables_IMAP[8], 2)} in politian factor.')

    # todo: calculate the individual preference scores for the results
    # Calculate individual preference scores for the results
    c1_res = calculate_cost(design_variables_IMAP[0], design_variables_IMAP[1], design_variables_IMAP[2], design_variables_IMAP[3], design_variables_IMAP[4], design_variables_IMAP[5], design_variables_IMAP[6], design_variables_IMAP[7], design_variables_IMAP[8])
    p1_res = pchip_interpolate(X_POINTS_COST, P_POINTS_COST, c1_res)

    c2_res = calculate_capacity(design_variables_IMAP[0], design_variables_IMAP[1], design_variables_IMAP[2], design_variables_IMAP[3], design_variables_IMAP[4], design_variables_IMAP[5], design_variables_IMAP[6], design_variables_IMAP[7], design_variables_IMAP[8])
    p2_res = pchip_interpolate(X_POINTS_CAPACITY, P_POINTS_CAPACITY, c2_res)

    c3_res = calculate_construction_time(design_variables_IMAP[0], design_variables_IMAP[1], design_variables_IMAP[2], design_variables_IMAP[3], design_variables_IMAP[4], design_variables_IMAP[5], design_variables_IMAP[6], design_variables_IMAP[7], design_variables_IMAP[8])  
    p3_res = pchip_interpolate(X_POINTS_ConsTime, P_POINTS_ConsTime, c3_res)

    c4_res = calculate_CO2_emissions(design_variables_IMAP[0], design_variables_IMAP[1], design_variables_IMAP[2], design_variables_IMAP[3], design_variables_IMAP[4], design_variables_IMAP[5], design_variables_IMAP[6], design_variables_IMAP[7], design_variables_IMAP[8])
    p4_res = pchip_interpolate(X_POINTS_CO2, P_POINTS_CO2, c4_res)

    c5_res = calculate_profit(design_variables_IMAP[0], design_variables_IMAP[1], design_variables_IMAP[2], design_variables_IMAP[3], design_variables_IMAP[4], design_variables_IMAP[5], design_variables_IMAP[6], design_variables_IMAP[7], design_variables_IMAP[8])
    p5_res = pchip_interpolate(X_POINTS_Profit, P_POINTS_Profit, c5_res)

    # Debugging prints to check calculated values
    print(f"c1_res: {c1_res}, p1_res: {p1_res}")
    print(f"c2_res: {c2_res}, p2_res: {p2_res}")
    print(f"c3_res: {c3_res}, p3_res: {p3_res}")
    print(f"c4_res: {c4_res}, p4_res: {p4_res}")
    print(f"c5_res: {c5_res}, p5_res: {p5_res}")

    # Plot the results on the preference curve subplots
    ax1.scatter(c1_res, p1_res, label='Optimal solution ' + paradigm[i], color=colours[i], marker=marker[i])
    ax2.scatter(c2_res, p2_res, label='Optimal solution ' + paradigm[i], color=colours[i], marker=marker[i])
    ax3.scatter(c3_res, p3_res, label='Optimal solution ' + paradigm[i], color=colours[i], marker=marker[i])
    ax4.scatter(c4_res, p4_res, label='Optimal solution ' + paradigm[i], color=colours[i], marker=marker[i])
    ax5.scatter(c5_res, p5_res, label='Optimal solution ' + paradigm[i], color=colours[i], marker=marker[i])

# Plot actual numbers in the projects
c1_Act = TotalCost
c2_Act = Capacity
c3_Act = ConstructionTime
c4_Act = TotalCO2
c5_Act = TotalCost * 0.2
p1_Act = pchip_interpolate(X_POINTS_COST, P_POINTS_COST, c1_Act)
p2_Act = pchip_interpolate(X_POINTS_CAPACITY, P_POINTS_CAPACITY, c2_Act)
p3_Act = pchip_interpolate(X_POINTS_ConsTime, P_POINTS_ConsTime, c3_Act)
p4_Act = pchip_interpolate(X_POINTS_CO2, P_POINTS_CO2, c4_Act)
p5_Act = pchip_interpolate(X_POINTS_Profit, P_POINTS_Profit, c5_Act)
ax1.scatter(c1_Act, p1_Act, label='Real life design', color=colours[2], marker=marker[2])
# ax2.scatter(c2_Act, p2_Act, label='Real life design', color=colours[2], marker=marker[2])
ax3.scatter(c3_Act, p3_Act, label='Real life design', color=colours[2], marker=marker[2])
# ax4.scatter(c4_Act, p4_Act, label='Real life design', color=colours[2], marker=marker[2])
# ax5.scatter(c5_Act, p5_Act, label='Real life design', color=colours[2], marker=marker[2])

# Plot calculated numbers in the projects
c1_Act = CalcCost
c2_Act = CalcCapacity
c3_Act = CalcConstructionTime
c4_Act = CalcCO2
c5_Act = CalcProfit
p1_Act = pchip_interpolate(X_POINTS_COST, P_POINTS_COST, c1_Act)
p2_Act = pchip_interpolate(X_POINTS_CAPACITY, P_POINTS_CAPACITY, c2_Act)
p3_Act = pchip_interpolate(X_POINTS_ConsTime, P_POINTS_ConsTime, c3_Act)
p4_Act = pchip_interpolate(X_POINTS_CO2, P_POINTS_CO2, c4_Act)
p5_Act = pchip_interpolate(X_POINTS_Profit, P_POINTS_Profit, c5_Act)
ax1.scatter(c1_Act, p1_Act, label='Calcu with real para', color=colours[3], marker=marker[3])
ax2.scatter(c2_Act, p2_Act, label='Calcu with real para', color=colours[3], marker=marker[3])
ax3.scatter(c3_Act, p3_Act, label='Calcu with real para', color=colours[3], marker=marker[3])
ax4.scatter(c4_Act, p4_Act, label='Calcu with real para', color=colours[3], marker=marker[3])
ax5.scatter(c5_Act, p5_Act, label='Calcu with real para', color=colours[3], marker=marker[3])

# Add legends and set titles for each subplot
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()

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

ax5.set_title('Optimal Solution for ' + str5 + ' (x5)')
ax5.set_xlabel(strTitleX5)
ax5.set_ylabel(strTitleY)

# Adjust the layout
fig.tight_layout()

# Display the plot
plt.show()