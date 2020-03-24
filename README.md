# Boltzmann analysis tool
This is a fitting tool for peaks consisting of four Lorentzian peaks which follow the *Boltzmann* distribution,
in especially Cm<sup>3+</sup> luminescence spectra.
## Getting started
You will need input data of xy format (tab, space or comma separated) and the following libraries:
* os
* numpy
* matplotlib
* tkinter
* lmfit

After changing the location of your input file and the temperature the data was collected at,
all you need to do is run the Boltzmann_fit_final2.py script.
## Testing
You can try out the functionality by using the Species1_25.txt file. Download the repository,
give the path to this very file in the script and you're ready to go. (T = 25°C in this example)