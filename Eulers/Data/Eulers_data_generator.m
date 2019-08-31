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
EulerDriver1D   

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
for i = 1:length(x)
    if mod(i,5) ~= 0
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
for i = 1:length(time_steps_storage)
    if mod(i,10) ~= 0
        temporal_indices_to_be_removed(vector_counter) = i;
        vector_counter = vector_counter + 1;
    end
end
t(temporal_indices_to_be_removed) = [];
rhosol(:,temporal_indices_to_be_removed) = [];
usol(:,temporal_indices_to_be_removed) = [];
Enersol(:,temporal_indices_to_be_removed) = [];

save('Abgrall_eulers','x','t','rhosol','usol','Enersol')

%% =======================================================================%
%                               Plotting
%=========================================================================%
% Fig_rhosol = figure;
% title('rho')
% 
% Fig_usol = figure;
% title('u')
% 
% Fig_Enersol = figure;
% title('Ener')
% 
% for tstep=1:length(t)
%     figure(Fig_rhosol);
%     plot(x,rhosol(:,tstep));
%     title('Density')
%     
%     pause(0.1)
% end
% 
% for tstep=1:length(t)
%     figure(Fig_usol)
%     plot(x,usol(:,tstep))
%     title('Velocity')
%     
%     pause(0.1)
% end
% 
% for tstep=1:length(t)
%     figure(Fig_Enersol)
%     plot(x,Enersol(:,tstep));
%     title('Ener')
%     
%     pause(0.1)
% end
