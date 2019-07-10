# beamCal
A Python module for beam parameter calculations at (HL-)collisions. Based on the work of S. Fartoukh.

## Description
Using the SciPy for numerical integration calculate the performance-related beam parameters (i.e. luminosity, luminous region, pileup etc) for a collider.

## Usage
```python
import beamCal as bc
x = bc.BeamCal()
x.setNpart0(1.2e11)
print(x.myLumi(0.15, 250.,0.))
```


## Contact
N. Karastathis, nkarast .at. cern .dot. ch
