#!/usr/bin/env python
from xtcav.GenerateLasingOffReference import *
GLOC=GenerateLasingOffReference();
GLOC.experiment='amon0816'
GLOC.runs='138'
GLOC.maxshots=8000
GLOC.nb=1
GLOC.islandsplitmethod = 'scipyLabel'
GLOC.groupsize=400
GLOC.SetValidityRange(138)
GLOC.Generate();
