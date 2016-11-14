#!/usr/bin/env python
from xtcav.GenerateLasingOffReference import *
GLOC=GenerateLasingOffReference();
GLOC.experiment='amon0816'
GLOC.runs='127'
GLOC.maxshots=3300
GLOC.nb=1
GLOC.islandsplitmethod = 'scipyLabel'
GLOC.groupsize=200
GLOC.SetValidityRange(127)
GLOC.Generate();
