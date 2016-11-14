#!/usr/bin/env python
from xtcav.GenerateLasingOffReference import *
GLOC=GenerateLasingOffReference();
GLOC.experiment='amon0816'
GLOC.runs='199'
GLOC.maxshots=8000
GLOC.nb=1
GLOC.islandsplitmethod = 'scipyLabel'
GLOC.groupsize=400
GLOC.SetValidityRange(192)
GLOC.Generate();
