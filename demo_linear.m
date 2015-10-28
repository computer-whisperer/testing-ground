function demo_linear
% A demo of iLQG/DDP with a control-limited LTI system.

% make stable linear dynamics
h = .01;        % time step
n = 3;         % state dimension
m = 2;          % control dimension
%A = randn(n,n);
%A = A-A';      % skew-symmetric = pure imaginary eigenvalues
%A = expm(h*A);  % discrete time
%B = h*randn(n,m);
A = [[1, -1.5e-4, -4.6e-5],
           [1.5e-4,  1,  1.1e-4],
           [4.5e-5, -1.1e-4,  1]];
B = [[-1.7e-5, 1.1e-4],
           [1.5e-4, 4.4e-6],
           [-1.4e-5, 3.7e-6]];

% quadratic costs
Q = h*eye(n);
R = .1*h*eye(m);

% control limits
Op.lims = ones(m,1)*[-1 1]*.6;
% Op.lims = [];

% optimization problem
DYNCST  = @(x,u,i) lin_dyn_cst(x,u,A,B,Q,R);
T       = 30;              % horizon
%x0      = randn(n,1);       % initial state
#u0      = .1*randn(m,T);    % initial controls

x0 = [[ 0.07919485],
      [-0.39150426],
      [ 0.39676904]];
u0 = [[0.08702516, -0.10695812, -0.03761507, 0.00790764, 0.00532442, 0.08218673, -0.04070015, 0.05781017, 0.0340251, 0.15567711],
      [-0.16786378, 0.08034461, -0.23664327, 0.16031643, 0.15972222, 0.00393588, -0.01797945, -0.14965136, 0.13926328, -0.00071236]];
u0 = repmat(u0, [1 T/10])
% run the optimization
[x, u, L] = iLQG(DYNCST, x0, u0, Op);
disp(L(:,:,T))

function [f,c,fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu] = lin_dyn_cst(x,u,A,B,Q,R)

% for a PD quadratic u-cost
% no cost (nans) is equivalent to u=0
u(isnan(u)) = 0;

if nargout == 2
    f = A*x + B*u;
    v11 = x.*(Q*x);
    v1 = sum(v11,1);
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