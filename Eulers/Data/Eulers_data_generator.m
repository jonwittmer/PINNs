close all
clear all
clc

addpath Codes1D
addpath ServiceRoutines
addpath CFD1D
addpath Grid/
addpath Grid/CFD
addpath Grid/Other

cpt = cputime;

% Compute Solution
EulerDriver1D   

% Plotting
x = x(:);
Fig_rho_tstep = figure;
title('rho')

Fig_u_tstep = figure;
title('u')

Fig_Ener_tstep = figure;
title('Ener')

for t=1:size(rho_tstep,2)
    figure(Fig_rho_tstep);
    plot(x,rho_tstep(:,t));
    
    pause(0.001)
end

for t=1:size(rho_tstep,2)
    figure(Fig_u_tstep)
    plot(x,u_tstep(:,t))
    title('u')
    
    pause(0.001)
end

for t=1:size(rho_tstep,2)
    figure(Fig_Ener_tstep)
    plot(x,Ener_tstep(:,t));
    title('Ener')
    
    pause(0.001)
end