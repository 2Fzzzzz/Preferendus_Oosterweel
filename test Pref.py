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

w1 = 0.3
w2 = 0.1
w3 = 0.2
w4 = 0.1
w5 = 0.3

X_POINTS_COST, P_POINTS_COST = [[300, 325, 600], [100, 60, 0]]
X_POINTS_CAPACITY, P_POINTS_CAPACITY = [[7337, 9500, 12692], [0, 50, 100]]
X_POINTS_ConsTime, P_POINTS_ConsTime = [[6.8, 12, 27.1], [100, 40, 0]]
X_POINTS_CO2, P_POINTS_CO2 = [[93.84, 173.0, 252], [100, 30, 0]]
X_POINTS_Profit, P_POINTS_Profit = [[60, 90, 120], [0, 40, 100]]

b1 = [1500, 2000]
b2 = [4, 6]
b3 = [5.5, 6.2]
b4 = [1.0, 1.6]
b5 = [2.8, 3.7]
b6 = [50, 100]
b8 = [1, 10]
b9 = [0.2, 1.0]
bounds = [b1, b2, b3, b4, b5, b6, b8, b9]

b1Len = 1800
b2Lanes = 6
b3Height = 5.8 
b4Thickness = 1.4
b5LaneWidth = 3.5
b6SpeedLimit = 80
b8Machines = 8
b9Politian = 1.0
TotalHeight = 10.0
TotalWidth = 42.0
ConstructionTime = 7.0
TotalCost = 500
TotalCO2 = 210
Capacity = 9300

start_Points_population = [b1Len, b2Lanes, b3Height, b4Thickness, b5LaneWidth, b6SpeedLimit, b8Machines, b9Politian]

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

def calculate_cost(x1, x2, x3, x4, x5, x6, x8, x9):
    return x1 * x2 * 0.05
def calculate_capacity(x1, x2, x3, x4, x5, x6, x8, x9):
    return 1620 * x2 * (1 + 0.022 * (x5 - 3.5)) * (1 + 0.003 * x6)
def calculate_construction_time(x1, x2, x3, x4, x5, x6, x8, x9):
    denom = np.maximum(x8 * x9, 1e-6)
    return (x1 * x2 / denom) ** 0.3
def calculate_CO2_emissions(x1, x2, x3, x4, x5, x6, x8, x9):
    return x1 * (x3 + 2 * x4 + 1) * (x2 * x5 + 2 * x4) * 0.0005
def calculate_profit(x1, x2, x3, x4, x5, x6, x8, x9):
    return calculate_cost(x1, x2, x3, x4, x5, x6, x8, x9) * 0.2

CalcCost = calculate_cost(b1Len, b2Lanes, b3Height, b4Thickness, b5LaneWidth, b6SpeedLimit, b8Machines, b9Politian)
CalcCapacity = calculate_capacity(b1Len, b2Lanes, b3Height, b4Thickness, b5LaneWidth, b6SpeedLimit, b8Machines, b9Politian)
CalcConstructionTime = calculate_construction_time(b1Len, b2Lanes, b3Height, b4Thickness, b5LaneWidth, b6SpeedLimit, b8Machines, b9Politian)
CalcCO2 = calculate_CO2_emissions(b1Len, b2Lanes, b3Height, b4Thickness, b5LaneWidth, b6SpeedLimit, b8Machines, b9Politian)
CalcProfit = calculate_profit(b1Len, b2Lanes, b3Height, b4Thickness, b5LaneWidth, b6SpeedLimit, b8Machines, b9Politian)

def objective_p1(x1, x2, x3, x4, x5, x6, x8, x9):
    return pchip_interpolate(X_POINTS_COST, P_POINTS_COST, (calculate_cost(x1, x2, x3, x4, x5, x6, x8, x9)))
def objective_p2(x1, x2, x3, x4, x5, x6, x8, x9):
    return pchip_interpolate(X_POINTS_CAPACITY, P_POINTS_CAPACITY, (calculate_capacity(x1, x2, x3, x4, x5, x6, x8, x9)))
def objective_p3(x1, x2, x3, x4, x5, x6, x8, x9):
    return pchip_interpolate(X_POINTS_ConsTime, P_POINTS_ConsTime, (calculate_construction_time(x1, x2, x3, x4, x5, x6, x8, x9)))
def objective_p4(x1, x2, x3, x4, x5, x6, x8, x9):
    return pchip_interpolate(X_POINTS_CO2, P_POINTS_CO2, (calculate_CO2_emissions(x1, x2, x3, x4, x5, x6, x8, x9)))
def objective_p5(x1, x2, x3, x4, x5, x6, x8, x9):
    return pchip_interpolate(X_POINTS_Profit, P_POINTS_Profit, (calculate_profit(x1, x2, x3, x4, x5, x6, x8, x9)))

def objective(variables):
    x1 = variables[:, 0]
    x2 = variables[:, 1]
    x3 = variables[:, 2]
    x4 = variables[:, 3]
    x5 = variables[:, 4]
    x6 = variables[:, 5]
    x8 = variables[:, 6]
    x9 = variables[:, 7]
    p_1 = objective_p1(x1, x2, x3, x4, x5, x6, x8, x9)
    p_2 = objective_p2(x1, x2, x3, x4, x5, x6, x8, x9)
    p_3 = objective_p3(x1, x2, x3, x4, x5, x6, x8, x9)
    p_4 = objective_p4(x1, x2, x3, x4, x5, x6, x8, x9)
    p_5 = objective_p5(x1, x2, x3, x4, x5, x6, x8, x9)
    return [w1, w2, w3, w4, w5], [p_1, p_2, p_3, p_4, p_5]

def constraint_min_thickness(variables):
    x3 = variables[:, 2]
    x4 = variables[:, 3]
    return 0.12 * x3 - x4
def constraint_max_thickness(variables):
    x3 = variables[:, 2]
    x4 = variables[:, 3]
    return x4 - 0.25 * x3
def constraint_aspect_ratio_min(variables):
    x2 = variables[:, 1]
    x3 = variables[:, 2]
    x5 = variables[:, 4]
    return 2.5 - (x2 * x5 / x3)
def constraint_aspect_ratio_max(variables):
    x2 = variables[:, 1]
    x3 = variables[:, 2]
    x5 = variables[:, 4]
    return (x2 * x5 / x3) - 5.5
def constraint_speed_width(variables):
    x5 = variables[:, 4]
    x6 = variables[:, 5]
    return x6 - (15 * x5 + 30)
def constraint_min_height(variables):
    x3 = variables[:, 2]
    return 5.5 - x3
def constraint_construction_equipment(variables):
    x1 = variables[:, 0]
    x8 = variables[:, 6]
    return (x1 / 300) - x8

cons = [['ineq', constraint_min_thickness], ['ineq', constraint_max_thickness], ['ineq', constraint_aspect_ratio_min], ['ineq', constraint_aspect_ratio_max], ['ineq', constraint_speed_width], ['ineq', constraint_min_height], ['ineq', constraint_construction_equipment]]

c1 = np.linspace(X_POINTS_COST[0], X_POINTS_COST[-1])
c2 = np.linspace(X_POINTS_CAPACITY[0], X_POINTS_CAPACITY[-1])
c3 = np.linspace(X_POINTS_ConsTime[0], X_POINTS_ConsTime[-1])
c4 = np.linspace(X_POINTS_CO2[0], X_POINTS_CO2[-1])
c5 = np.linspace(X_POINTS_Profit[0], X_POINTS_Profit[-1])

p1 = pchip_interpolate(X_POINTS_COST, P_POINTS_COST, (c1))
p2 = pchip_interpolate(X_POINTS_CAPACITY, P_POINTS_CAPACITY, (c2))
p3 = pchip_interpolate(X_POINTS_ConsTime, P_POINTS_ConsTime, (c3))
p4 = pchip_interpolate(X_POINTS_CO2, P_POINTS_CO2, (c4))
p5 = pchip_interpolate(X_POINTS_Profit, P_POINTS_Profit, (c5))

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

fig.savefig("Oosterweel.png")

paradigm = ['minmax', 'tetra']
marker = ['o', '*', 's', '^']
colours = ['orange', 'green', 'red', 'blue']

fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)

ax1.plot(c1, p1, label='Preference curve', color='black')
ax2.plot(c2, p2, label='Preference curve', color='black')
ax3.plot(c3, p3, label='Preference curve', color='black')
ax4.plot(c4, p4, label='Preference curve', color='black')
ax5.plot(c5, p5, label='Preference curve', color='black')

for i in range(2):
    options = {
        'n_bits': 8,
        'n_iter': 1000000,
        'n_pop': 50,
        'r_cross': 0.8,
        'max_stall': 100000,
        'aggregation': paradigm[i],
        "var_type_mixed": ["int", "int", "real", "real", "real", "int", "real", "int", "real"],
    }
    print(f'Run GA with {paradigm[i]}')
    ga = GeneticAlgorithm(objective=objective, constraints=cons, bounds=bounds, options=options, start_points_population=[start_Points_population])
    score_IMAP, design_variables_IMAP, _ = ga.run()
    print(f'Optimal method {paradigm[i]}, result for x1 = {round(design_variables_IMAP[0], 2)} meters in length and '
          f'x2 = {round(design_variables_IMAP[1], 2)} lanes and '
          f'x3 = {round(design_variables_IMAP[2], 2)} meters in height and '
          f'x4 = {round(design_variables_IMAP[3], 2)} meters in thickness and '
          f'x5 = {round(design_variables_IMAP[4], 2)} meters in lane width and '
          f'x6 = {round(design_variables_IMAP[5], 2)} km/h in speed limit and '
          f'x8 = {round(design_variables_IMAP[6], 2)} machines and '
          f'x9 = {round(design_variables_IMAP[7], 2)} in politian factor.')
    c1_res = calculate_cost(design_variables_IMAP[0], design_variables_IMAP[1], design_variables_IMAP[2], design_variables_IMAP[3], design_variables_IMAP[4], design_variables_IMAP[5], design_variables_IMAP[6], design_variables_IMAP[7])
    p1_res = pchip_interpolate(X_POINTS_COST, P_POINTS_COST, c1_res)
    c2_res = calculate_capacity(design_variables_IMAP[0], design_variables_IMAP[1], design_variables_IMAP[2], design_variables_IMAP[3], design_variables_IMAP[4], design_variables_IMAP[5], design_variables_IMAP[6], design_variables_IMAP[7])
    p2_res = pchip_interpolate(X_POINTS_CAPACITY, P_POINTS_CAPACITY, c2_res)
    c3_res = calculate_construction_time(design_variables_IMAP[0], design_variables_IMAP[1], design_variables_IMAP[2], design_variables_IMAP[3], design_variables_IMAP[4], design_variables_IMAP[5], design_variables_IMAP[6], design_variables_IMAP[7])  
    p3_res = pchip_interpolate(X_POINTS_ConsTime, P_POINTS_ConsTime, c3_res)
    c4_res = calculate_CO2_emissions(design_variables_IMAP[0], design_variables_IMAP[1], design_variables_IMAP[2], design_variables_IMAP[3], design_variables_IMAP[4], design_variables_IMAP[5], design_variables_IMAP[6], design_variables_IMAP[7])
    p4_res = pchip_interpolate(X_POINTS_CO2, P_POINTS_CO2, c4_res)
    c5_res = calculate_profit(design_variables_IMAP[0], design_variables_IMAP[1], design_variables_IMAP[2], design_variables_IMAP[3], design_variables_IMAP[4], design_variables_IMAP[5], design_variables_IMAP[6], design_variables_IMAP[7])
    p5_res = pchip_interpolate(X_POINTS_Profit, P_POINTS_Profit, c5_res)
    print(f"c1_res: {c1_res}, p1_res: {p1_res}")
    print(f"c2_res: {c2_res}, p2_res: {p2_res}")
    print(f"c3_res: {c3_res}, p3_res: {p3_res}")
    print(f"c4_res: {c4_res}, p4_res: {p4_res}")
    print(f"c5_res: {c5_res}, p5_res: {p5_res}")
    ax1.scatter(c1_res, p1_res, label='Optimal solution ' + paradigm[i
