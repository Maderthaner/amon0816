#!/usr/bin/env python
from xtcav.GenerateDarkBackground import *
GDB=GenerateDarkBackground();
GDB.experiment='amon0816'
GDB.runs='137'
GDB.maxshots=1000
GDB.SetValidityRange(137)
GDB.Generate();
