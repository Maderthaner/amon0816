#!/usr/bin/env python
from xtcav.GenerateDarkBackground import *
GDB=GenerateDarkBackground();
GDB.experiment='amon0816'
GDB.runs='89'
GDB.maxshots=1000
GDB.SetValidityRange(1)
GDB.Generate();
