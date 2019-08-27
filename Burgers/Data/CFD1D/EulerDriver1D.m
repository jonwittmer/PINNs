% Driver script for solving the 1D Euler equations
Globals1D;

% Polynomial order used for approximation 
N = 6;

% Generate simple mesh
[Nv, VX, K, EToV] = MeshGen1D(0.0, 1.0, 250);

% Initialize solver and construct grid and metric
StartUp1D;
gamma = 1.4;

% Set up initial conditions -- Sod's problem
MassMatrix = inv(V')/V;
cx = ones(Np,1)*sum(MassMatrix*x,1)/2; 

% rho = ones(Np,K).*( (cx<0.5) + 0.125*(cx>=0.5));
% rhou = zeros(Np,K);
% Ener = ones(Np,K).*((cx<0.5) + 0.1*(cx>=0.5))/(gamma-1.0);

% Sod's Problem - Abgrall
rho_sod = ones(Np,K).*((cx<=0.5) + 0.125*(cx>0.5));
u_sod = zeros(Np,K);
p_sod = ones(Np,K).*((cx<=0.5) + 0.1*(cx>0.5));

% Lax's Problem - Abgrall
rho_lax = ones(Np,K).*(0.445*(cx<=0.5) + 0.5*(cx>0.5));
u_lax = ones(Np,K).*(0.698*(cx<=0.5) + 0*(cx>0.5));
p_lax = ones(Np,K).*(3.528*(cx<=0.5) + 0.571*(cx>0.5));

mu = 0.6;
rho = mu*rho_lax + (1-mu)*rho_sod;
u = mu*u_lax + (1-mu)*u_sod;
p = mu*p_lax + (1-mu)*p_sod;
Ener = p./(gamma-1.0) + (1/2)*rho.*u.^2;

FinalTime = 0.2;

% Solve Problem
[rho,rhou,Ener] = Euler1D(rho,rhou,Ener,FinalTime);
