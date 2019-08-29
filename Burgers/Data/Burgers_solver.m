%This code below was obtained and modified from: https://math.stackexchange.com/questions/2763316/the-method-of-characteristics-for-burgers-equation

mx = 500; % number of nodes in x
CFL = 0.95; % Courant number
g = @(x) (1-cos(x)).*(0<=x).*(x<=2*pi); % initial condition
tend = 1.3; % final time

% initialization
t = 0;
x = linspace(0,2*pi,mx)';
dx = (x(end)-x(1))/(mx-1);
u = g(x);
utemp = u;
dt = CFL*dx/max(u);

figure;
hch = plot(x+t*g(x), g(x), 'k--');
hold on
hlf = plot(x, g(x), 'b.');
xlabel('x');
ylabel('u');
xlim([x(1) x(end)]);
ylim([0 2.1]);
ht = title(sprintf('t = %0.2f',t));

while (t+dt<tend)
    % Lax-Friedrichs
    for i=2:mx-1
        dflux = 0.5*u(i+1)^2 - 0.5*u(i-1)^2;
        utemp(i) = 0.5*(u(i+1) + u(i-1)) - 0.5*dt/dx* dflux;
    end
    utemp(1) = utemp(mx-1);
    utemp(mx) = utemp(2);

    u = utemp;
    t = t + dt;
    dt = CFL*dx/max(u);
    set(hch,'XData',x+t*g(x));
    set(hlf,'YData',u);
    set(ht,'String',sprintf('t = %0.2f',t));
    drawnow;
end
legend('Char.','LF','Location','northwest');