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

mu = 0.65;
usol = zeros(size(x,1),size(t,1));

%============================================================================================================
% This code below was obtained and modified from: https://math.stackexchange.com/questions/2763316/the-method-of-characteristics-for-burgers-equation

mx = size(x,1); % number of nodes in x
CFL = 0.95; % Courant number
initial_condition = @(x) (mu*abs(sin(2*x))+0.1).*(0<=x).*(x<=pi); % initial condition
tend = t(end); % final time

%=== Initialization ===%
current_time = 0;
dx = (x(end)-x(1))/(mx-1);
u = initial_condition(x);
utemp = u;
% dt = CFL*dx/max(u);
dt = temporal_domain_finaltime/n_timesteps;

figure;
hch = plot(x+current_time*initial_condition(x), initial_condition(x), 'k--');
hold on
hlf = plot(x, initial_condition(x), 'b.');
xlabel('x');
ylabel('u');
xlim([x(1) x(end)]);
ylim([0 0.7]);
ht = title(sprintf('t = %0.2f',current_time));

%=== Numerical Solver ===%
for time=1:size(t,1)
    for i=2:mx-1
        dflux = 0.5*u(i+1)^2 - 0.5*u(i-1)^2;
        utemp(i) = 0.5*(u(i+1) + u(i-1)) - 0.5*dt/dx* dflux;
    end
    utemp(1) = utemp(mx-1); % periodic boundary conditions
    utemp(mx) = utemp(2);   % periodic boundary conditions

    u = utemp;
    usol(:,time) = u;
    current_time = t(time);
%     dt = CFL*dx/max(u);
    set(hch,'XData',x+current_time*initial_condition(x));
    set(hlf,'YData',u);
    set(ht,'String',sprintf('t = %0.2f',current_time));
    drawnow;
end
legend('Char.','LF','Location','northwest');

close all

%=== Checking Solution ===%
for time=1:size(t,1)
    plot(x,usol(:,time))
    xlim([x(1) x(end)]);
    ylim([0 0.7]);
    pause(0.01)
end

save('Abgrall_burgers_shock','x','t','usol')