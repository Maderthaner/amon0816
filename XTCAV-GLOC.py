#!/usr/bin/env python
from xtcav.GenerateLasingOffReference import *
GLOC=GenerateLasingOffReference();
GLOC.experiment='amon0816'
GLOC.runs='89'
GLOC.maxshots=500
GLOC.nb=2
GLOC.islandsplitmethod = 'scipyLabel'
GLOC.groupsize=180
GLOC.SetValidityRange(1)
GLOC.Generate();
