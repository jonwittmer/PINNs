%% GAS DYNAMICS
%% RIEMANN PROBLEM
% Numerical solution of the Riemann problem with initial conditions
% piecewise constant. Problem shock tube in which at the time t=0- two states 
% have been defined: u1, a1, p1 and u4, a4, p4. At the time t=0+ a septum 
% separating the two regions is raised impulsively.
% We determine the states 2 and 3, debating between the 4 possible types
% solution : RCR , RCS , SCR, SCS .
% In particular, if the initial data are such as to induce a type of
% solution of the type NCR , NCS, RCN , SCN, then the program provides
% such a type and it is possible to deduct from the results, whereby at
% example the speed of sound of three contiguous states is equal , as long as
% the input of initial data having an accuracy up to the digit
% decimal significant error.
% Example: state 1: u1=0.00000000000 a1=1.00000000000 p1= 1.00000000000
%          state 4: u4=1.94747747000 a4=1.38949549000 p4=10.00000000000
% The solution is NCR , but the program could provide SCR if the
% datas are not precise enough to a suitable digit , however, is
% known as the "shock" on u-a really is nothing in intensity , it is as if
% there wasn't , in fact you get a Mach its upstream ( and downstream ) of
% to 1 . And yet you will have a1 = a2 = a3 . So be careful and to correlate the
% type of solution with the correct interpretation of the results !
% WE STRONGLY RECOMMEND TO ENTER DATA WITH AN INITIAL ACCURACY
% UP TO THE AMOUNT REPORTED SIGNIFICANT DECIMAL ERROR.
% NOTATION:
% u     = gas velocity [m/s]
% p     = gas pressure [Pa]
% a     = sound velocity [m/s]
% gamma = ratio of the specific heats (cp/cv)
% rho   = gas density [kg/m�]
clear all
close all
clc
format long
%% -----------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%% Initial Conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sod's Problem - Abgrall
rho_sod_left = 1;
u_sod_left = 0;
p_sod_left = 1;
rho_sod_right = 0.125;
u_sod_right = 0;
p_sod_right = 0.1;

% Lax's Problem - Abgrall
rho_lax_left = 0.445;
u_lax_left = 0.698;
p_lax_left = 3.528;
rho_lax_right = 0.5;
u_lax_right = 0;
p_lax_right = 0.571;

mu = 0; % Linear combination parameter

% Initial Condition
rho_left = mu*rho_sod_left + (1-mu)*rho_lax_left;
u_left = mu*u_sod_left + (1-mu)*u_lax_left;
p_left = mu*p_sod_left + (1-mu)*p_lax_left;

rho_right = mu*rho_sod_right + (1-mu)*rho_lax_right;
u_right = mu*u_sod_right + (1-mu)*u_lax_right;
p_right = mu*p_sod_right + (1-mu)*p_lax_right;

p1 = p_left;
u1 = u_left;
gamma1 = 1.4;
rho1 = rho_left;
a1 = sqrt(gamma1*p1/rho1);

p4 = p_right;
u4 = u_right;
gamma4 = 1.4;
rho4 = rho_right;
a4 = sqrt(gamma4*p4/rho4);
e = 1e-5;
%% -------------------------------------------------------------------------
delta1 = 0.5*(gamma1-1);
delta4 = 0.5*(gamma4-1);
disp('');
%% ------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%% Admissibility Solution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Condition of the existence of the solution');
esist = (u4-u1 <= a4/delta4+a1/delta1);
if (esist)
    disp('OK! The problem admits solution');
else
    disp('NO! The problem doesn''t admit solution with the used method');
end
%% -----------------------------------------------------------------------
%%%%%%%%%%% Value for the first atempt u0 and computation %%%%%%%%%%%%%%%%%
if (esist)
    disp('');
    if (p1<p4)
        sigma = gamma4;
    else
        sigma = gamma1;
    end
    z = ((gamma1-1)/(gamma4-1))*(a4/a1)*((p1/p4)^((sigma-1)/(2*sigma)));
    u0 = (z*(a1+delta1*u1)-(a4-delta4*u4))/((1+z)*(sigma-1)/2);
    disp('Valore di primo tentativo per u0. [m/s] u0 = '),disp(u0);
    
    u = u0;
    p2 = 1;
    p3 = 0.5;
    k = 0;
    
    while (abs(1-p2/p3)>e)
        if (u<u1)
            b = ((gamma1+1)/4)*((u1-u)/a1);
            M1_rel = b + sqrt(b^2+1);
            w1 = u1-M1_rel*a1;
            M1_abs = u1/a1;
            p2 = p1*(1+(2*gamma1/(gamma1+1))*(M1_rel^2-1));
            dp2 = -2*gamma1*(p1/a1)*((abs(M1_rel))^3/(1+M1_rel^2));
            a2 = a1*sqrt((gamma1+1+(gamma1-1)*(p2/p1))/(gamma1+1+(gamma1-1)*(p1/p2)));
            M2_rel = (u-w1)/a2;
            M2_abs = u/a2;
        else
            a2 = a1-delta1*(u-u1);
            p2 = p1*(a2/a1)^(gamma1/delta1);
            dp2 = -gamma1*(p2/a2);
            M2_abs = u/a2;
            M1_abs = u1/a1;
        end
        if (u>u4)
            b = ((gamma1+1)/4)*((u4-u)/a4);
            M4_rel = b - sqrt(b^2+1);
            w4 = u4-M4_rel*a4;
            M4_abs = u4/a4;
            p3 = p4*(1+(2*gamma4/(gamma4+1))*(M4_rel^2-1));
            dp3 = 2*gamma4*(p4/a4)*((abs(M4_rel))^3/(1+M4_rel^2));
            a3 = a4*sqrt((gamma4+1+(gamma4-1)*(p3/p4))/(gamma4+1+(gamma4-1)*(p4/p3)));
            M3_rel = (u-w4)/a3;
            M3_abs = u/a3;
        else
            a3 = a4+delta4*(u-u4);
            p3 = p4*(a3/a4)^(gamma4/delta4);
            dp3 = gamma4*(p3/a3);
            M3_abs = u/a3;
            M4_abs = u4/a4;
        end
        u = u - (p2-p3)/(dp2-dp3);
        k = k + 1;
    end
    
    rho2 = gamma1*(p2/a2^2);
    rho3 = gamma4*(p3/a3^2);
    
    clc 
    
    %---------------------------------------------------------------------
    
    %% Type Solution
    
    if ((u>u1)&&(abs(p2-p1)>e)&&(abs(p2-p4)>e))
        if (u<u4)
            tip_sol = 'RCR';
        else
            tip_sol = 'RCS';
        end
    elseif ((u<u1)&&(abs(p2-p1)>e)&&(abs(p2-p4)>e))
        if (u<u4)
            tip_sol = 'SCR';
        else
            tip_sol = 'SCS';
        end
    elseif ((abs(u-u1)<e)&&(abs(p2-p1)<e)&&(abs(u-u4)>e))
        if (u<u4)
            tip_sol = 'NCR';
            u = u1;
            p2 = p1; p3 = p1;
            a2 = a1; a3 = a1;
            rho2 = rho1; rho3 = rho1;
            M2_abs = M1_abs; M3_abs = M1_abs;            
        else
            tip_sol = 'NCS';
            u = u1;
            p2 = p1; p3 = p1;
            a2 = a1; a3 = a1;
            rho2 = rho1; rho3 = rho1;
            M2_abs = M1_abs; M3_abs = M1_abs;
            M2_rel = M1_rel; M3_rel = M1_rel;
            w1 = w4;
        end
    elseif ((abs(u-u1)>e)&&(abs(p2-p4)<e)&&(abs(u-u4)<e))
        if (u>u1)
            tip_sol = 'RCN';
            u = u4;
            p2 = p4; p3 = p4;
            a2 = a4; a3 = a4;
            rho2 = rho4; rho3 = rho4;
            M2_abs = M4_abs; M3_abs = M4_abs;
        else
            tip_sol = 'SCN';
            u = u4;
            p2 = p4; p3 = p4;
            a2 = a4; a3 = a4;
            rho2 = rho4; rho3 = rho4;
            M2_abs = M4_abs; M3_abs = M4_abs;
            M2_rel = M4_rel; M3_rel = M4_rel;
            w4 = w1;
        end
    else
        tip_sol = 'NCN';
        u = u1;
        p2 = p1; p3 = p1;
        a2 = a1; a3 = a1;
        rho2 = rho1; rho3 = rho1;
        M2_abs = M1_abs; M3_abs = M1_abs;        
    end 
      
    %---------------------------------------------------------------------
    
    %% View of the results
    
    disp('Type Solution: '),disp(tip_sol);
    disp('');
    disp('Iterations = '),disp(k);
    disp(''),disp('--------------------------------------------------------------');
    disp('state 1.');
    disp(' u1   [m/s]   = '),disp(u1);
    disp(' p1   [Pa]    = '),disp(p1);
    disp(' a1   [m/s]   = '),disp(a1);
    disp(' rho1 [kg/m�] = '),disp(rho1);
    disp(' M1_abs       = '),disp(M1_abs);
    if (strcmp(tip_sol,'NCS')||strcmp(tip_sol,'SCS')||strcmp(tip_sol,'SCN')||strcmp(tip_sol,'SCR'))
        disp(' M1_rel       = '),disp(M1_rel);
        disp(' w    [m/s]   = '),disp(w1);
    end
    disp(''),disp('--------------------------------------------------------------');
    disp('state 2.');
    disp(' u2   [m/s]   = '),disp(u);
    disp(' p2   [Pa]    = '),disp(p2);
    disp(' a2   [m/s]   = '),disp(a2);
    disp(' rho2 [kg/m�] = '),disp(rho2);
    disp(' M2_abs       = '),disp(M2_abs);
    if (strcmp(tip_sol,'NCS')||strcmp(tip_sol,'SCS')||strcmp(tip_sol,'SCN')||strcmp(tip_sol,'SCR'))
        disp(' M2_rel       = '),disp(M2_rel);
        disp(' w    [m/s]   = '),disp(w1);
    end
    disp(''),disp('--------------------------------------------------------------');
    disp('state 3.');
    disp(' u3   [m/s]   = '),disp(u);
    disp(' p3   [Pa]    = '),disp(p3);
    disp(' a3   [m/s]   = '),disp(a3);
    disp(' rho3 [kg/m�] = '),disp(rho3);
    disp(' M3_abs       = '),disp(M3_abs);
    if (strcmp(tip_sol,'NCS')||strcmp(tip_sol,'SCS')||strcmp(tip_sol,'SCN')||strcmp(tip_sol,'RCS'))
        disp(' M3_rel       = '),disp(M3_rel);
        disp(' w    [m/s]   = '),disp(w4);
    end
    disp(''),disp('--------------------------------------------------------------');
    disp('state 4.');
    disp(' u4   [m/s]   = '),disp(u4);
    disp(' p4   [Pa]    = '),disp(p4);
    disp(' a4   [m/s]   = '),disp(a4);
    disp(' rho4 [kg/m�] = '),disp(rho4);
    disp(' M4_abs       = '),disp(M4_abs);
    if (strcmp(tip_sol,'NCS')||strcmp(tip_sol,'SCS')||strcmp(tip_sol,'SCN')||strcmp(tip_sol,'RCS'))
        disp(' M4_rel       = '),disp(M4_rel);
        disp(' w    [m/s]   = '),disp(w4);
    end
end
u2 = u;
u3 = u;
