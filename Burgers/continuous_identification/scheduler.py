#!/usr/bin/env python3
'''
    author = Jon Wittmer
    
    This code provides a scheduler to run a variety of different 
    parameter configurations for the Burgers equation
'''
from Burgers_ADMM import Parameters
import nvidia_smi
import copy
import subprocess
import os
from mpi4py import MPI
from time import sleep

class FLAGS:
    RECEIVED = 1
    RUN_FINISHED = 2
    EXIT = 3
    NEW_RUN = 4

class Scheduler:
    def __init__(self, params, total_gpus = 4):
        self.FLAGS = FLAGS()

        self.total_gpus = total_gpus
        
        nvidia_smi.nvmlInit()

        params_list = [params.N_u, params.N_f, params.rho, params.epochs]
        self.scenarios_list = self.Assemble_Parameters(params_list)
        
        # create list of parameters for calling
        self.scenarios = []
        for vals in self.scenarios_list:
            p        = Parameters()
            p.N_u    = vals[0]
            p.N_f    = vals[1]
            p.rho    = vals[2]
            p.epochs = vals[3]
            self.scenarios.append(copy.deepcopy(p))

    def Assemble_Parameters(self, params):
        # params is a list of lists, with each inner list representing
        # a different model parameter. This function constructs the combinations
        return self.Get_Combinations(params[0], params[1:])
        
    def Get_Combinations(self, params1, params2):
        # assign here in case this is the last list item
        combos = params2[0]
        
        # reassign when it is not the last item - recursive algorithm
        if len(params2) > 1:
            combos = self.Get_Combinations(params2[0], params2[1:])
            
        # concatenate the output into a list of lists
        output = []
        for i in params1:
            for j in combos:
                # convert to list if not already
                j = j if isinstance(j, list) else [j]
                
                # for some reason, this needs broken into 3 lines...Python
                temp = [i]
                temp.extend(j)
                output.append(temp)
                
        return output

    def Schedule_Runs(self, comm):
        scenarios_left = len(self.scenarios)
        print(str(scenarios_left) + ' total runs')
        
        # initialize available processes
        available_processes = [1, 2, 3, 4]

        # start running tasks
        while scenarios_left > 0:
            
            # check for returning processes
            s = MPI.Status()
            comm.Iprobe(status=s)
            if s.tag == self.FLAGS.RUN_FINISHED:
                print('Run ended. Starting new thread.')
                data = comm.recv()
                scenarios_left -= 1
                if len(self.scenarios) == 0:
                    comm.send([], s.source, self.FLAGS.EXIT)
                else: 
                    available_processes.append(s.source)

            # assign training to process
            available_gpus = self.Available_GPUs()
            print(available_gpus)
            print(available_processes)

            if len(available_gpus) > 0 and len(available_processes) > 0 and len(self.scenarios) > 0:
                curr_process = available_processes.pop(0)
                curr_scenario = self.scenarios.pop(0)
                curr_scenario.gpu = str(available_gpus.pop(0))
                
                print('Beginning Training of NN:')
                self.Print_Scenario(curr_scenario)
                print()
                
                # block here to make sure the process starts before moving on so we don't overwrite buffer
                print('current process: ' + str(curr_process))
                req = comm.isend(curr_scenario, curr_process, self.FLAGS.NEW_RUN)
                req.wait()
                
            elif len(available_processes) > 0 and len(self.scenarios) == 0:
                while len(available_processes) > 0:
                    proc = available_processes.pop(0)
                    comm.send([], proc, self.FLAGS.EXIT)

            sleep(30)
                
        
    def Available_GPUs(self):
        available = []
        for i in range(self.total_gpus):
            handle  = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            res     = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            if res.gpu < 30 and (mem_res.used / mem_res.total *100) < 30:
                available.append(i)
        return available 

    def Print_Scenario(self, p):
        print()
        print(f'    p.N_u:    {p.N_u}')
        print(f'    p.N_f:    {p.N_f}')
        print(f'    p.rho:    {p.rho}')
        print(f'    p.epochs: {p.epochs}')
        print(f'    p.gpu:    {p.gpu}')
        print()

if __name__ == '__main__':
    # mpi stuff
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        
        params = Parameters()
        params.N_u = [100, 200]
        params.N_f = [100, 200, 500, 1000]#, 5000, 10000]
        params.rho = [1.0, 10.0, 50.0, 100.0]
        params.epochs = [1e6]#, 1e5, 5e5, 1e6]
        
        sched = Scheduler(params)
        
        print()
        print(sched.Available_GPUs())
        
        sched.Schedule_Runs(comm)
    
    else:
        while True:
            status = MPI.Status()
            data = comm.recv(source=0, status=status)
            
            if status.tag == FLAGS.EXIT:
                break
            
            proc = subprocess.Popen(['./launch_NN.py', f'{data.N_u}', f'{data.N_f}', f'{data.rho}', f'{int(data.epochs)}', f'{data.gpu}'])
            proc.wait()
            
            req = comm.isend([], 0, FLAGS.RUN_FINISHED)
            req.wait()
            
            
