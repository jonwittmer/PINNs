#!/usr/bin/env python3

import sys
from Burgers_batch_L2 import Parameters, PhysicsInformedNN

p = Parameters()
p.N_u = int(sys.argv[1])
p.N_f = int(sys.argv[2])
p.epochs = int(sys.argv[3])
p.gpu = str(sys.argv[4])

var = PhysicsInformedNN(p)
