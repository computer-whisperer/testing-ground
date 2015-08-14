function demo_linear
% A demo of iLQG/DDP with a control-limited LTI system.

% make stable linear dynamics
h = .01;        % time step
n = 10;         % state dimension
m = 2;          % control dimension
A = randn(n,n);
A = A-A';       % skew-symmetric = pure imaginary eigenvalues
A = expm(h*A);  % discrete time
B = h*randn(n,m);

% quadratic costs
Q = h*eye(n);
R = .1*h*eye(m);

% control limits
Op.lims = ones(m,1)*[-1 1]*.6;

% optimization problem
DYNCST  = @(x,u,i) lin_dyn_cst(x,u,A,B,Q,R);
T       = 10;              % horizon
x0      = randn(n,1);       % initial state
u0      = .1*randn(m,T);    % initial controls

% run the optimization
iLQG(DYNCST, x0, u0, Op);


function [f,c,fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu] = lin_dyn_cst(x,u,A,B,Q,R)

% for a PD quadratic u-cost
% no cost (nans) is equivalent to u=0
u(isnan(u)) = 0;

if nargout == 2
    f = A*x + B*u;
    v1 = sum(x.*(Q*x),1);
    v2 = sum(u.*(R*u),1);
    c = 0.5*v1 + 0.5*v2;
else
    N   = size(x,2);
    fx  = repmat(A, [1 1 N]);
    fu  = repmat(B, [1 1 N]);
    cx  = Q*x;
    cu  = R*u;
    cxx = repmat(Q, [1 1 N]);
    cxu = repmat(zeros(size(B)), [1 1 N]);
    cuu = repmat(R, [1 1 N]);
    [f,c,fxx,fxu,fuu] = deal([]);
end