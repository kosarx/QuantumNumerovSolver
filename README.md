# QuantumNumerovSolver
----------
An application in Python Numpy that uses the Numerov-Cooley Method to numerically solve the 1-D Time-independent Schrödinger Equation.

## Numerical Methods
----------
The Numerov Method is given by:
![The Numerov Method Formula](images/numerovMethod.png)

Cooley's Energy Correction Formula is:
![Cooley's Energy Correction Formula](images/CooleysEnergyCorrectionFormula.png)

## Showcase
----------

### Particle In A Box
![Particle in a Box with Quantum Number 1 and 100 points](images/PIBn1N100.png)
![Particle in a Box with Quantum Number 8 and 400 points](images/PIBn8N400.png)

### Finite Potential Square Well
![Finite Potential Square Well with Quantum Number 1 and 100 points](images/FPWn1N100.png)
![Finite Potential Square Well with Quantum Number 4 and 200 points](images/FPWn4N200.png)

### The Quantum Harmonic Oscillator
![Quantum Harmonic Oscillator with Quantum Number 1 and 100 points](images/QHOn1N100.png)
![Quantum Harmonic Oscillator with Quantum Number 2 and 100 points](images/QHOn2N100.png)
![Quantum Harmonic Oscillator with Quantum Number 3 and 100 points](images/QHOn3N100.png)
![Quantum Harmonic Oscillator with Quantum Number 6 and 200 points](images/QHOn6N200.png)

### Pöschl-Teller Potential
![Pöschl-Teller Potential with Quantum Number 1 and 100 points](images/PTWn1N100.png)
![Pöschl-Teller Potential with Quantum Number 4 and 100 points](images/PTWn4N100.png)

### Double Finite Potential Well
![Double Finite Potential Well with Quantum Number 1 and 100 points](images/DWPn1N100.png)
![Double Finite Potential Well with Quantum Number 2 and 500 points](images/DWPn2N500.png)
![Double Finite Potential Well with Quantum Number 3 and 100 points](images/DWPn3N100.png)
![Double Finite Potential Well with Quantum Number 4 and 100 points](images/DWPn4N100.png)
![Uneven Double Finite Potential Well with Quantum Number 3 and 100 points](images/unevenDWPn3N100.png)
![No Barrier Double Finite Potential Well with Quantum Number 3 and 100 points](images/noBarrierDWPn3N100.png)

# How To:
----------
Save to a seperate folder and simply run main.py. 

## Requires
---------- 
Python 3, numpy, scipy, and matplotlib.

# Discussion
----------
I highly recommend that you read QuantumNumerovSolver.pdf. It gives a detailed rundown of the numerical methods used, as well as a brief overview of the python code, with some expected
results. 

This has been developed for University of Patras, Department of Electrical and Computer Engineering during the course of Introduction to Quantum Electronics, under Prof. Emmanuel Paspalakis.

My main goal in uploading this is to help out any other poor student that struggled as much as me to find some kind of implementation of these numerical methods in anything other than FORTRAN90. As a
result of this, the program and code developed is of dubious quality, with questionable results (details in the pdf) for anything other than undergraduate coursework. Still, it prints some nice wavefunctions
for the Particle In a Box Problem, for a Finite Potential Well, for the Quantum Harmonic Oscillator, a Pöschl-Teller Potential and a Double Finite Potential Well, all while using the Numerov Method, with 
matching, and Cooley's Energy Correction Formula for a given quantum state.

Special mention must go out to Joshua Izaac and Jingbo Wang with their book Computational Quantum Mechanics, which basically carried the entire numerical side of this work, as well as to the awesome interactive
lesson over at Liu Lab Computational Chemistry Tutorials (https://liu-group.github.io/interactive-Numerov-PIB/), which basically carried the python/numpy side of this work.
Many other sources went into this as well, which are mentioned at the pdf.
