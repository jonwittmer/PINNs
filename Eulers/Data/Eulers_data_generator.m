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

%% =======================================================================%
%                             Compute Solution
%=========================================================================%
mu = 0.6; % Linear combination parameter
EulerDriver1D   
save('Abgrall_eulers_accurate_solution','x','time_steps_storage','rho_tstep','u_tstep','Ener_tstep')

%% =======================================================================%
%                                Outputs
%=========================================================================%
rhosol = rho_tstep;
usol = u_tstep;
Enersol = Ener_tstep;
t = time_steps_storage';

%=== Removing Duplicate Spatial Data ===%
x = x(:);
[~, ind] = unique(x); % identify indices for unique entries of x
duplicate_ind = setdiff(1:size(x, 1), ind); % identify indices for duplicate indices of x
x(duplicate_ind) = [];
rhosol(duplicate_ind,:) = [];
usol(duplicate_ind,:) = [];
Enersol(duplicate_ind,:) = [];

%=== Removing Unwanted Spatial Data ===%
vector_counter = 1;
for i = 2:length(x)
    if mod(i,3) ~= 0
        spatial_indices_to_be_removed(vector_counter) = i;
        vector_counter = vector_counter + 1;
    end
end
x(spatial_indices_to_be_removed,:) = [];
rhosol(spatial_indices_to_be_removed,:) = [];
usol(spatial_indices_to_be_removed,:) = [];
Enersol(spatial_indices_to_be_removed,:) = [];

%=== Removing Unwanted Temporal Data ===%
vector_counter = 1;
for i = 2:length(time_steps_storage)
    if mod(i,3) ~= 0
        temporal_indices_to_be_removed(vector_counter) = i;
        vector_counter = vector_counter + 1;
    end
end
t(temporal_indices_to_be_removed) = [];
rhosol(:,temporal_indices_to_be_removed) = [];
usol(:,temporal_indices_to_be_removed) = [];
Enersol(:,temporal_indices_to_be_removed) = [];

save('Abgrall_eulers','x','t','rhosol','usol','Enersol')

fprintf('Solution saved')

%% =======================================================================%
%                               Plotting
%=========================================================================%
Fig_rhosol = figure;
Fig_usol = figure;
% Fig_Enersol = figure;
Fig_psol = figure;

for tstep=1:length(t)
    figure(Fig_rhosol);
    plot(x,rhosol(:,tstep));
    title('Density')
    
    pause(0.1)
end

for tstep=1:length(t)
    figure(Fig_usol)
    plot(x,usol(:,tstep))
    title('Velocity')
    
    pause(0.1)
end

% for tstep=1:length(t)
%     figure(Fig_Enersol)
%     plot(x,Enersol(:,tstep));
%     title('Energy')
%     
%     pause(0.1)
% end

gamma = 1.4;
psol = (gamma - 1)*(Enersol - (1/2)*rhosol.*(usol.^2));

for tstep=1:length(t)
    figure(Fig_psol)
    plot(x,psol(:,tstep));
    title('Pressure')
    
    pause(0.1)
end
