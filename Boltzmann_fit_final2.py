"""
Takes in Cm emission asc files and tries to fit a designated region
with gaussian peaks following the Boltzmann distribution.

Input: x,y data of spectrum
Ouput: x, y, accumlative peak fit, lorentz peaks 1-4

Version: 1.0
Creator: Manuel Eibl
"""

import numpy as np
from math import pi
import lmfit
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog


# defines how the input data is read
def load_data(inp_data):
    xdata = []
    ydata = []
    with open(inp_data, 'r') as f:
        data = f.readlines()
    for line in data:
        try:
            splitted_line = line.split()
            x = float(splitted_line[0])
            y = float(splitted_line[1])
            xdata.append(x)
            ydata.append(y)
        except:
            try:
                splitted_line = line.split(',')
                x = float(splitted_line[0])
                y = float(splitted_line[1])
                xdata.append(x)
                ydata.append(y)
            except:
                pass
    return xdata, ydata


# normalization of data
def norm_data(data):
    out_norm_data = []
    for value in data:
        out_value = value/max(data)
        out_norm_data.append(out_value)
    return out_norm_data


# Function to calculate the state population relative to the lowest ground state
def boltzmann(lambda1, lambda2, temperature):
    k = 1.38064852 * 10 ** -23  # J/K
    planck = 6.62607015 * 10 ** -34  # J*s
    c = 299792458  # m/s
    T = 273.15 + temperature  # K
    return np.exp(planck*c*((1/(lambda1*10**-9)) - (1/(lambda2*10**-9)))/(k*T))


# lorentzian peak definition
def lorentzian(x, amplitude, center, sigma, offset):
    return amplitude/pi*(sigma/((x - center)**2 + sigma**2)) + offset


# this function calculates the amplitude derived from the boltzmann equation
def amplitude_calculator(c1, c2, c3, c4, a1, s1, s2, s3, s4, T):

    # This is in case the order of the peaks has changed during fitting
    value_dict = {}
    sigma_list = [s1, s2, s3, s4]
    for i, element in enumerate([c1, c2, c3, c4]):
        dict_key = int(element*100)
        value_dict[dict_key] = sigma_list[i]
    ordered_centers = sorted([c1, c2, c3, c4])
    c1 = ordered_centers[3]
    c2 = ordered_centers[2]
    c3 = ordered_centers[1]
    c4 = ordered_centers[0]

    # calculates boltzmann distribution
    boltzmann_ratio2 = boltzmann(c1, c2, T)
    boltzmann_ratio3 = boltzmann(c1, c3, T)
    boltzmann_ratio4 = boltzmann(c1, c4, T)

    # calculates the integrals derived from the boltzmann distribution
    # integral1 = integral_lorentzian([a, b], a1, c1, s1, offset)
    a2 = a1*boltzmann_ratio2
    a3 = a1*boltzmann_ratio3
    a4 = a1*boltzmann_ratio4

    return a2, a3, a4


# adds four lorentzian peaks; needed for plotting
def four_lorentzian(x, a1, c1, s1, a2, c2, s2, a3, c3, s3, a4, c4, s4, offset):
    return (lorentzian(x, a1, c1, s1, offset=0) +
            lorentzian(x, a2, c2, s2, offset=0) +
            lorentzian(x, a3, c3, s3, offset=0) +
            lorentzian(x, a4, c4, s4, offset=0) + offset)


# takes the fitting parameters and adds four lorentzian peaks; needed later in least square fitting
def four_lorentzian_fit(x, pars):
    vals = pars.valuesdict()
    a1 = vals['a1']
    c1 = vals['c1']
    s1 = vals['s1']
    a2 = vals['a2']
    c2 = vals['c2']
    s2 = vals['s2']
    a3 = vals['a3']
    c3 = vals['c3']
    s3 = vals['s3']
    a4 = vals['a4']
    c4 = vals['c4']
    s4 = vals['s4']
    offset = vals['offset']
    return (lorentzian(x, a1, c1, s1, offset=0) +
            lorentzian(x, a2, c2, s2, offset=0) +
            lorentzian(x, a3, c3, s3, offset=0) +
            lorentzian(x, a4, c4, s4, offset=0) + offset)


def error_func(pars, x, y):
    return (four_lorentzian_fit(x, pars) - y)**2


# A first fit with the requirements of using four lorentzian functions; Starting values are given below
def inital_fit(xdata, ydata):
    pars = lmfit.Parameters()
    # The following parameters are the initial guess and can be changed
    # Must have: value=xxx, the rest is optional
    # Values can be fixed with '...value=xxx, vary=False), this is, however, not recommended for initial fit
    pars.add('a1', value=0.5, min=0)
    pars.add('c1', value=600, max=xdata[-1])
    pars.add('s1', value=2, min=0)
    pars.add('a2', value=0.5, min=0)
    pars.add('c2', value=596, max=pars.valuesdict()['c1'])
    pars.add('s2', value=2, min=0)
    pars.add('a3', value=0.5, min=0)
    pars.add('c3', value=592, max=pars.valuesdict()['c2'])
    pars.add('s3', value=2, min=0)
    pars.add('a4', value=0.5, min=0)
    pars.add('c4', value=586, min=xdata[0], max=pars.valuesdict()['c3'])
    pars.add('s4', value=2, min=1)
    pars.add('offset', value=0.03)

    init_out = lmfit.minimize(fcn=error_func, params=pars, args=(xdata, ydata))     # starts the fitting algorithm

    # creating the fitting output
    init_out_dict = {}
    for name, param in init_out.params.items():
        init_out_dict[name] = param.value

    print(lmfit.fit_report(init_out))

    # plotting the fitting results
    plt.plot(xdata, ydata, 'bo', label='data', markersize=4)
    plt.plot(xdata,
             four_lorentzian(xdata, init_out_dict['a1'], init_out_dict['c1'], init_out_dict['s1'], init_out_dict['a2'],
                             init_out_dict['c2'],
                             init_out_dict['s2'], init_out_dict['a3'], init_out_dict['c3'], init_out_dict['s3'],
                             init_out_dict['a4'],
                             init_out_dict['c4'], init_out_dict['s4'], init_out_dict['offset']), '--r', label='fit')
    plt.plot(xdata,
             lorentzian(xdata, init_out_dict['a1'], init_out_dict['c1'], init_out_dict['s1'], init_out_dict['offset']),
             label='lorentz1')
    plt.plot(xdata,
             lorentzian(xdata, init_out_dict['a2'], init_out_dict['c2'], init_out_dict['s2'], init_out_dict['offset']),
             label='lorentz2')
    plt.plot(xdata,
             lorentzian(xdata, init_out_dict['a3'], init_out_dict['c3'], init_out_dict['s3'], init_out_dict['offset']),
             label='lorentz3')
    plt.plot(xdata,
             lorentzian(xdata, init_out_dict['a4'], init_out_dict['c4'], init_out_dict['s4'], init_out_dict['offset']),
             label='lorentz4')
    plt.legend()
    #plt.ion()
    plt.show()

    fit_ok = input('Is this initial fit ok? (\'n\' for no) ')
    if fit_ok == 'n':
        return
    return init_out_dict


# Actual fitting using the boltzmann distribution as prerequisist
def boltzmann_peak_fit(xdata, ydata, init_out_dict, temperature):
    # Calculate the amplitudes in accordance with the Boltzmann equation
    a2_val, a3_val, a4_val = amplitude_calculator(init_out_dict['c1'], init_out_dict['c2'], init_out_dict['c3'],
                                                  init_out_dict['c4'], init_out_dict['a1'], init_out_dict['s1'],
                                                  init_out_dict['s2'], init_out_dict['s3'], init_out_dict['s4'],
                                                  temperature)

    pars = lmfit.Parameters()
    # Fix all parameters for first fit
    pars.add('a1', value=init_out_dict['a1'], vary=False)
    pars.add('c1', value=init_out_dict['c1'], vary=False)
    pars.add('s1', value=init_out_dict['s1'], min=0)
    pars.add('a2', value=a2_val, vary=False)
    pars.add('c2', value=init_out_dict['c2'], vary=False)
    pars.add('s2', value=init_out_dict['s2'], min=0, max=3*pars.valuesdict()['s1'])
    pars.add('a3', value=a3_val, vary=False)
    pars.add('c3', value=init_out_dict['c3'], vary=False)
    pars.add('s3', value=init_out_dict['s3'], min=0, max=3*pars.valuesdict()['s2'])
    pars.add('a4', value=a4_val, vary=False)
    pars.add('c4', value=init_out_dict['c4'], vary=False)
    pars.add('s4', value=init_out_dict['s4'], min=0, max=2*pars.valuesdict()['s3'])
    pars.add('offset', value=init_out_dict['offset'])

    out = lmfit.minimize(fcn=error_func, params=pars, args=(xdata, ydata))     # starts the fitting algorithm

    # creating the fitting output
    out_dict = {}
    for name, param in out.params.items():
        out_dict[name] = param.value

    # plotting the fitting results
    plt.plot(xdata, ydata, 'bo', label='data', markersize=4)
    plt.plot(xdata,
             four_lorentzian(xdata, out_dict['a1'], out_dict['c1'], out_dict['s1'], out_dict['a2'], out_dict['c2'],
                             out_dict['s2'], out_dict['a3'], out_dict['c3'], out_dict['s3'], out_dict['a4'],
                             out_dict['c4'], out_dict['s4'], out_dict['offset']), '--r', label='fit')
    plt.plot(xdata, lorentzian(xdata, out_dict['a1'], out_dict['c1'], out_dict['s1'], out_dict['offset']),
             label='lorentz1')
    plt.plot(xdata, lorentzian(xdata, out_dict['a2'], out_dict['c2'], out_dict['s2'], out_dict['offset']),
             label='lorentz2')
    plt.plot(xdata, lorentzian(xdata, out_dict['a3'], out_dict['c3'], out_dict['s3'], out_dict['offset']),
             label='lorentz3')
    plt.plot(xdata, lorentzian(xdata, out_dict['a4'], out_dict['c4'], out_dict['s4'], out_dict['offset']),
             label='lorentz4')
    plt.legend()
    #plt.ion()
    print(lmfit.fit_report(out))
    print('First fitting result after setting the Boltzmann distribution')
    plt.show()

    i = 0

    # Now a few fitting cycles where the amplitudes and sigmas are fitted
    # followed by one where the amplitudes are recalculated using the Boltzmann equation
    # and then everything fixed

    while i < 20:    # number of cycles
        a2_val, a3_val, a4_val = amplitude_calculator(init_out_dict['c1'], init_out_dict['c2'], init_out_dict['c3'],
                                                      init_out_dict['c4'], init_out_dict['a1'], init_out_dict['s1'],
                                                      init_out_dict['s2'], init_out_dict['s3'], init_out_dict['s4'],
                                                      temperature)

        pars.add('a1', value=out_dict['a1'], vary=False)
        pars.add('c1', value=out_dict['c1'], max=xdata[-1])
        pars.add('s1', value=out_dict['s1'])
        pars.add('a2', value=a2_val, vary=False)
        pars.add('c2', value=out_dict['c2'])
        pars.add('s2', value=out_dict['s2'])
        pars.add('a3', value=a3_val, vary=False)
        pars.add('c3', value=out_dict['c3'])
        pars.add('s3', value=out_dict['s3'])
        pars.add('a4', value=a4_val, vary=False)
        pars.add('c4', value=out_dict['c4'])
        pars.add('s4', value=out_dict['s4'], vary=False)
        pars.add('offset', value=out_dict['offset'])

        out = lmfit.minimize(fcn=error_func, params=pars, args=(xdata, ydata))     # starts the fitting algorithm
        out_dict = {}
        for name, param in out.params.items():
            out_dict[name] = param.value

        print(lmfit.fit_report(out))

        i +=1

    a2_val, a3_val, a4_val = amplitude_calculator(init_out_dict['c1'], init_out_dict['c2'], init_out_dict['c3'],
                                                  init_out_dict['c4'], init_out_dict['a1'], init_out_dict['s1'],
                                                  init_out_dict['s2'], init_out_dict['s3'], init_out_dict['s4'],
                                                  temperature)

    pars.add('a1', value=out_dict['a1'], vary=False)
    pars.add('c1', value=out_dict['c1'], vary=False)
    pars.add('s1', value=out_dict['s1'], vary=False)
    pars.add('a2', value=a2_val, vary=False)
    pars.add('c2', value=out_dict['c2'], vary=False)
    pars.add('s2', value=out_dict['s2'], vary=False)
    pars.add('a3', value=a3_val, vary=False)
    pars.add('c3', value=out_dict['c3'], vary=False)
    pars.add('s3', value=out_dict['s3'], vary=False)
    pars.add('a4', value=a4_val, vary=False)
    pars.add('c4', value=out_dict['c4'], vary=False)
    pars.add('s4', value=out_dict['s4'], vary=False)
    pars.add('offset', value=out_dict['offset'], vary=False)

    out = lmfit.minimize(fcn=error_func, params=pars, args=(xdata, ydata))     # starts the fitting algorithm
    out_dict = {}
    for name, param in out.params.items():
        out_dict[name] = param.value

    print(lmfit.fit_report(out))

    # plotting the fitting results
    plt.plot(xdata, ydata, 'bo', label='data', markersize=4)
    plt.plot(xdata,
             four_lorentzian(xdata, out_dict['a1'], out_dict['c1'], out_dict['s1'], out_dict['a2'], out_dict['c2'],
                             out_dict['s2'], out_dict['a3'], out_dict['c3'], out_dict['s3'], out_dict['a4'],
                             out_dict['c4'], out_dict['s4'], out_dict['offset']), '--r', label='fit')
    plt.plot(xdata, lorentzian(xdata, out_dict['a1'], out_dict['c1'], out_dict['s1'], out_dict['offset']),
             label='lorentz1')
    plt.plot(xdata, lorentzian(xdata, out_dict['a2'], out_dict['c2'], out_dict['s2'], out_dict['offset']),
             label='lorentz2')
    plt.plot(xdata, lorentzian(xdata, out_dict['a3'], out_dict['c3'], out_dict['s3'], out_dict['offset']),
             label='lorentz3')
    plt.plot(xdata, lorentzian(xdata, out_dict['a4'], out_dict['c4'], out_dict['s4'], out_dict['offset']),
             label='lorentz4')
    plt.legend()
    #plt.ion()
    print(lmfit.fit_report(out))
    plt.show()

    return lmfit.fit_report(out), out_dict


def output_maker(xdata, out_dict, T):

    output = ''
    out_fit_data = four_lorentzian(xdata, out_dict['a1'], out_dict['c1'], out_dict['s1'], out_dict['a2'],
                                   out_dict['c2'], out_dict['s2'], out_dict['a3'], out_dict['c3'], out_dict['s3'],
                                   out_dict['a4'], out_dict['c4'], out_dict['s4'], out_dict['offset'])
    ydata_lorentz1 = lorentzian(xdata, out_dict['a1'], out_dict['c1'], out_dict['s1'], out_dict['offset'])
    ydata_lorentz2 = lorentzian(xdata, out_dict['a2'], out_dict['c2'], out_dict['s2'], out_dict['offset'])
    ydata_lorentz3 = lorentzian(xdata, out_dict['a3'], out_dict['c3'], out_dict['s3'], out_dict['offset'])
    ydata_lorentz4 = lorentzian(xdata, out_dict['a4'], out_dict['c4'], out_dict['s4'], out_dict['offset'])

    output_params = (str(fit_report) + '\n' + 'Area(Lorentz1) = ' + str(out_dict['a1']) + '\n' +
                     'Boltzmann area(Lorentz2) = ' + str(boltzmann_lorentz2) + '\n' +
                     'Area(Lorentz2) = ' + str(out_dict['a2']) + '\n' +
                     'Boltzmann area(Lorentz3) = ' + str(boltzmann_lorentz3) + '\n' +
                     'Area(Lorentz3) = ' + str(out_dict['a3']) + '\n' +
                     'Boltzmann area(Lorentz4) = ' + str(boltzmann_lorentz4) + '\n' +
                     'Area(Lorentz4) = ' + str(out_dict['a4']))

    output += ('Wavelength / nm' + '\t' + 'Raw data' + '\t' + 'Cumulative peak fit' + '\t' + 'Lorentz 1' +
               '\t' + 'Lorentz 2' + '\t' + 'Lorentz 3' + '\t' + 'Lorentz 4' + '\n')
    for i, value in enumerate(xdata):
        output += (str(value) + '\t' + str(ydata[i]) + '\t' + str(out_fit_data[i]) + '\t' +
                   str(ydata_lorentz1[i]) + '\t' + str(ydata_lorentz2[i]) + '\t' + str(ydata_lorentz3[i]) + '\t' +
                   str(ydata_lorentz4[i]) + '\n')

    return output, output_params


def save_file(output, output_params):

    output_path = input('Where would you like to have your output file? ')
    output_file = input('Output name: ')
    output_data_file = output_file + '.asc'
    output_param_file = output_file + '_param.asc'

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    abs_FILE_output = os.path.join(output_path, output_data_file)
    abs_PARAM_output = os.path.join(output_path, output_param_file)

    while os.path.exists(abs_FILE_output):
        window = tk.Tk()
        window.withdraw()
        if tk.messagebox.askyesno('Overwrite warning',
                                  f'File {output_file} already exists.\nWould you like to overwrite it?') is True:
            os.remove(abs_FILE_output)
            if os.path.exists(abs_PARAM_output):
                os.remove(abs_PARAM_output)

        else:
            new_name = tk.simpledialog.askstring('input string', 'New Filename:')
            new_out_name = new_name + '.asc'
            new_param_name = new_name + '_param.asc'
            abs_FILE_output = os.path.join(output_path, new_out_name)
            abs_PARAM_output = os.path.join(output_path, new_param_name)

        window.deiconify()
        window.destroy()
        window.quit()

    with open(str(abs_FILE_output), 'w+') as o:
        o.write(str(output))
    with open(str(abs_PARAM_output), 'w+') as o:
        o.write(str(output_params))


###################################################################
# Data select and processing
###################################################################
inp_data = r'e:\Python\TRLFS\Boltzmann_fitting\Species1_25.txt'     # give path to data file
xdata, ydata = load_data(inp_data)

T = 25  #in °C

norm = '1'  # '1' = normalize to maximum, anything else is no fit (e.g. '0')
if norm == '1':
    ydata = norm_data(ydata)

# create a numpy array from list of x and y data
xdata = np.array(xdata)
ydata = np.array(ydata)

###################################################################
# Run script
###################################################################
if __name__ == '__main__':
    init_out_dict = inital_fit(xdata, ydata)

    if init_out_dict is not None:
        fit_report, out_dict = boltzmann_peak_fit(xdata, ydata, init_out_dict, T)

    ###################################################################
    # Output of integral comparison
    ###################################################################
    boltzmann2_final = boltzmann(out_dict['c1'], out_dict['c2'], T)
    boltzmann3_final = boltzmann(out_dict['c1'], out_dict['c3'], T)
    boltzmann4_final = boltzmann(out_dict['c1'], out_dict['c4'], T)

    boltzmann_lorentz2 = out_dict['a1']*boltzmann2_final
    boltzmann_lorentz3 = out_dict['a1']*boltzmann3_final
    boltzmann_lorentz4 = out_dict['a1']*boltzmann4_final

    print('Area(Lorentz1) = ' + str(out_dict['a1']))
    print('Boltzmann area(Lorentz2) = ' + str(boltzmann_lorentz2))
    print('Area(Lorentz2) = ' + str(out_dict['a2']))
    print('Boltzmann area(Lorentz3) = ' + str(boltzmann_lorentz3))
    print('Area(Lorentz3) = ' + str(out_dict['a3']))
    print('Boltzmann area(Lorentz4) = ' + str(boltzmann_lorentz4))
    print('Area(Lorentz4) = ' + str(out_dict['a4']))

    ###################################################################
    # Data output
    ###################################################################
    save = input("Wanna save these results?['y' for yes] ")
    yes = ['y', 'yes', '1']
    if save in yes:
        output, output_params = output_maker(xdata, out_dict, T)
        save_file(output, output_params)