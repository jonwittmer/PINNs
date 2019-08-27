close all
clear all
clc

%addpath Codes1D
addpath ServiceRoutines
addpath CFD1D
addpath Grid/
addpath Grid/CFD
addpath Grid/Other

cpt = cputime;

EulerDriver1D   
figure; 
plot(x,rho); 

figure
plot(x,rhou)

figure 
plot(x,Ener);