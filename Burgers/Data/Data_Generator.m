close all
clear all
clc

%=== Spatial Domain ===%
spatial_domain_leftboundary = 0;
spatial_domain_rightboundary = pi;
n_gridpoints = 256;
x = spatial_domain_leftboundary:(spatial_domain_rightboundary/n_gridpoints):spatial_domain_rightboundary;
x = x';

%=== Time Domain ===%
temporal_domain_starttime = 0;
temporal_domain_finaltime = pi;
n_timesteps = 256;
t = temporal_domain_starttime:(temporal_domain_finaltime/n_timesteps):temporal_domain_finaltime;
t = t';

%=== True Solution ===%
usol = zeros(size(x,1),size(t,1));
mu = 0.575;
wavespeed = 0.5*mu + 0.1;

for i = 1:size(x,1)
    for j = 1:size(t,1)
        usol(i,j) = initial_condition(x(i),t(j),mu,wavespeed);
    end
end

%=== Plotting True Solution ===%
for i=1:size(usol,2)
    plot(x,usol(:,i))
    pause(0.1)
end

% save('Abgrall_burgers_shock')


%=========================================================================
function u = initial_condition(x,t,mu,wavespeed)

u = mu*abs(sin(2*x - wavespeed*t)) + 0.1;

end