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

%=== Compute Solution ===%
EulerDriver1D   

%=== Removing Duplicate Unwanted Spatial Data ===%
spatial_indices_to_be_removed = [2,3,5,6];
x(spatial_indices_to_be_removed,:) = [];
x = x(:);
[~, ind] = unique(x); % identify indices for unique entries of x
duplicate_ind = setdiff(1:size(x, 1), ind); % indentify indices for duplicate indices of x
x(duplicate_ind) = [];
rho_tstep(duplicate_ind,:) = [];
u_tstep(duplicate_ind,:) = [];
Ener_tstep(duplicate_ind,:) = [];

%=== Removing Unwanted Temporal Data ===%
vector_counter = 1;
for i = 1:length(time_steps_storage)
    if mod(i,10) ~= 0
        temporal_indices_to_be_removed(vector_counter) = i;
        vector_counter = vector_counter + 1;
    end
end
t = time_steps_storage';
t(temporal_indices_to_be_removed) = [];

%=== Outputs ===%
rhosol = rho_tstep(:);
usol = u_tstep(:);
Enersol = Ener_tstep(:);

save('Abgrall_eulers','x','t','rhosol','usol','Enersol')

clear rho_tstep u_tstep Ener_tstep times_steps_storage

%=== Plotting ===%
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
%     
%     pause(0.01)
% end
% 
% for tstep=1:length(t)
%     figure(Fig_usol)
%     plot(x,usol(:,tstep))
%     title('u')
%     
%     pause(0.01)
% end
% 
% for tstep=1:length(t)
%     figure(Fig_Enersol)
%     plot(x,Enersol(:,tstep));
%     title('Ener')
%     
%     pause(0.01)
% end
