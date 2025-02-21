function [rho_tstep,u_tstep,Ener_tstep,time_steps_storage] = Euler1D(rho, rhou, Ener, FinalTime, BC)

% function [rho, rhou, Ener] = Euler1D(rho, rhou, Ener, FinalTime, BC)
% Purpose  : Integrate 1D Euler equations until FinalTime starting with
%            initial conditions [rho, rhou, Ener]

Globals1D;

% Parameters
gamma = 1.4; CFL = 1.0; time = 0;

% Prepare for adaptive time stepping
mindx = min(x(2,:)-x(1,:));

% Limit initial solution
rho =SlopeLimitN(rho); rhou=SlopeLimitN(rhou); Ener=SlopeLimitN(Ener);

time_counter = 1; % For storing solutions at each time step

% outer time step loop 
while(time<FinalTime)
  
  Temp = (Ener - 0.5*(rhou).^2./rho)./rho;
  cvel = sqrt(gamma*(gamma-1)*Temp);
  dt = CFL*min(min(mindx./(abs(rhou./rho)+cvel)));
  
  if(time+dt>FinalTime)
    dt = FinalTime-time;
  end

  % 3rd order SSP Runge-Kutta
  
  % SSP RK Stage 1.
  [rhsrho,rhsrhou,rhsEner]  = EulerRHS1D(rho, rhou, Ener, BC);
  rho1  = rho  + dt*rhsrho;
  rhou1 = rhou + dt*rhsrhou;
  Ener1 = Ener + dt*rhsEner;

  % Limit fields
  rho1  = SlopeLimitN(rho1); rhou1 = SlopeLimitN(rhou1); Ener1 = SlopeLimitN(Ener1);

  % SSP RK Stage 2.
  [rhsrho,rhsrhou,rhsEner]  = EulerRHS1D(rho1, rhou1, Ener1, BC);
  rho2   = (3*rho  + rho1  + dt*rhsrho )/4;
  rhou2  = (3*rhou + rhou1 + dt*rhsrhou)/4;
  Ener2  = (3*Ener + Ener1 + dt*rhsEner)/4;

  % Limit fields
  rho2  = SlopeLimitN(rho2); rhou2 = SlopeLimitN(rhou2); Ener2 = SlopeLimitN(Ener2);

  % SSP RK Stage 3.
  [rhsrho,rhsrhou,rhsEner]  = EulerRHS1D(rho2, rhou2, Ener2,BC);
  rho  = (rho  + 2*rho2  + 2*dt*rhsrho )/3;
  rhou = (rhou + 2*rhou2 + 2*dt*rhsrhou)/3;
  Ener = (Ener + 2*Ener2 + 2*dt*rhsEner)/3;

  % Limit solution
  rho = SlopeLimitN(rho); rhou=SlopeLimitN(rhou); Ener=SlopeLimitN(Ener);
  
  rho_tstep(:,time_counter) = rho(:);
  u_tstep(:,time_counter) = rhou(:)./rho(:);
  Ener_tstep(:,time_counter) = Ener(:);
  
  % Increment time and adapt timestep
  time_steps_storage(time_counter) = time;
  time = time+dt;
  time_counter = time_counter + 1;
  
end
return
