#!/usr/bin/env python3

import sys
from Burgers_ADMM import Parameters, PINNWrapper

p = Parameters()
p.N_u = int(sys.argv[1])
p.N_f = int(sys.argv[2])
p.rho = float(sys.argv[3])
p.epochs = int(sys.argv[4])
p.gpu = str(sys.argv[5])

var = PINNWrapper(p)
