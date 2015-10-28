from numpy import *
from ilqg import ilqg
# A demo of iLQG/DDP with car-parking dynamics




## ==== graphics ====
#
## prepare the axes
#figure(4)
#clf
#set(gca,'xlim',[-4 4],'ylim',[-4 4],'DataAspectRatio',[1 1 1])
#grid on
#box on
#
## plot target configuration with light colors
#h      = car_plot([0 0 0 0]', [0 0]')
#fcolor = get(h,'facecolor')
#ecolor = get(h,'edgecolor')
#fcolor = cellfun(@(x) (x+3)/4,fcolor,'UniformOutput',false)
#ecolor = cellfun(@(x) (x+3)/4,ecolor,'UniformOutput',false)
#set(h, {'facecolor','edgecolor'}, [fcolor ecolor])
#
## animate the trajectory
#h = []
#for i=1:T
#   delete(h)
#   h = car_plot(x(:,i), u(:,i))
#   drawnow    


def car_dynamics(x, u):
    
    # === states and controls:
    # x = [x y t v]' = [x y car_angle front_wheel_velocity]
    # u = [w a]'     = [front_wheel_angle accelaration]
    
    # constants
    d  = 2.0      # d = distance between back and front axles
    h  = 0.03     # h = timestep (seconds)
    
    # controls
    w  = u[0,:,:] # w = front wheel angle
    a  = u[1,:,:] # a = front wheel acceleration
    
    o  = x[2,:,:] # o = car angle
                  # z = unit_vector(o)
    z  = [cos(o), sin(o)] 
    
    v  = x[3,:,:] # v = front wheel velocity
    f  = h*v      # f = front wheel rolling distance
                   # b = back wheel rolling distance
    b  = d + f*cos(w) - sqrt(d^2 - (f*sin(w))**2)
                   # do = change in car angle
    dod = asin(sin(w)*f/d)
    
    dy = [tt(b, z), dod, h*a]   # change in state
    y  = x + dy                # new state
    return y


def car_cost(x, u):
    # cost function for car-parking problem
    # sum of 3 terms:
    # lu: quadratic cost on controls
    # lf: final cost on distance from target parking configuration
    # lx: small running cost on distance from origin to encourage tight turns
    
    final = isnan(u[1,:])
    u[:,final]  = 0
    
    cu  = 1e-2*[1, .01]         # control cost coefficients
    
    cf  = [ .1,  .1,   1,  .3]    # final cost coefficients
    pf  = [.01, .01, .01,  1].conj().T    # smoothness scales for final cost
    
    cx  = 1e-3*[1,  1]          # running cost coefficients
    px  = [.1, .1].conj().T             # smoothness scales for running cost
    
    # control cost
    lu    = cu*u**2
    
    # final cost
    if any(final):
       llf      = cf*sabs(x[:,final],pf)
       lf       = real(final)
       lf[final]= llf
    else:
       lf    = 0
    
    
    # running cost
    lx = cx*sabs(x[1:2,:],px)
    
    # total const
    c     = lu + lx + lf
    return c

def sabs(x,p):
    # smooth absolute-value function (a.k.a pseudo-Huber)
    y = pp( sqrt(pp(x**2,p**2)), -p)
    return y


def car_dyn_cst(x, u, full_DDP, want_all=False):
    #function [f,c,fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu] = car_dyn_cst(x,u,full_DDP)
    # combine car dynamics and cost
    # use helper function finite_difference() to compute derivatives
    
    if not want_all:
        f = car_dynamics(x,u)
        c = car_cost(x,u)
        return f, c
    else:
        
        # dynamics derivatives
        xu_dyn  = lambda xu: car_dynamics(xu[1:4,:], xu[5:6,:])
        J       = finite_difference(xu_dyn, [x, u])
        fx      = J[:,1:4,:]
        fu      = J[:,5:6,:]
        
        # cost first derivatives
        xu_cost = lambda xu: car_cost(xu[1:4,:],xu[5:6,:])
        J       = squeeze(finite_difference(xu_cost, [x, u]))
        cx      = J[1:4, :]
        cu      = J[5:6, :]
        
        # cost second derivatives
        xu_Jcst = lambda xu: squeeze(finite_difference(xu_cost, xu))
        JJ      = finite_difference(xu_Jcst, [x, u])
        cxx     = JJ[1:4,1:4,:]
        cxu     = JJ[1:4,5:6,:]
        cuu     = JJ[5:6,5:6,:]
        
        # dynamics second derivatives
        if full_DDP:
            xu_Jcst = lambda xu: finite_difference(xu_dyn, xu)
            JJ      = finite_difference(xu_Jcst, [x, u])
            JJ      = reshape(JJ, [4, 6, size(J)])
            JJ      = 0.5*(JJ + permute(JJ,[1, 3, 2, 4]))
            fxx     = JJ[:,1:4,1:4,:]
            fxu     = JJ[:,1:4,5:6,:]
            fuu     = JJ[:,5:6,5:6,:]
        else:
            fxx, fxu, fuu = deal([])
        
        return fx, fu, fxx, fxu, fuu, cx, cu, cxx, cxu, cuu



def finite_difference(fun, x, h=None):
    # simple finite-difference derivatives
    # assumes the function fun() is vectorized
    
    if h is None:
        h = 2^-17
    
    
    [n, K]  = size(x)
    H       = [zeros(n,1), h*eye(n)]
    H       = permute(H, [1, 3, 2])
    X       = pp(x, H)
    X       = reshape(X, n, K*(n+1))
    Y       = fun(X)
    m       = numel(Y)/(K*(n+1))
    Y       = reshape(Y, m, K, n+1)
    J       = pp(Y[:,:,2:], -Y[:,:,1]) / h
    J       = permute(J, [1, 3, 2])

# Set full_DDP=true to compute 2nd order derivatives of the
# dynamics. This will make iterations more expensive, but
# final convergence will be much faster (quadratic)
full_DDP = True

# optimization problem
DYNCST  = lambda x, u, i: car_dyn_cst(x,u,full_DDP)
T       = 500              # horizon
x0      = array([1, 1, pi*3/2, 0])   # initial state
u0      = .1*random.randn(2,T)  # initial controls
options = {}
options["lims"]  = array([[-.5, .5],         # wheel angle limits (radians)
                          [ -2,  2]])       # acceleration limits (m/s^2)

# run the optimization
options["maxIter"] = 5
x, u = ilqg(DYNCST, x0, u0, options)

## ======== graphics functions ========
#function h = car_plot(x,u)
#
#body        = [0.9 2.1 0.3]           # body = [width length curvature]
#bodycolor   = 0.5*[1 1 1]
#headlights  = [0.25 0.1 .1 body(1)/2] # headlights [width length curvature x]
#lightcolor  = [1 1 0]
#wheel       = [0.15 0.4 .06 1.1*body(1) -1.1 .9]  # wheels = [width length curvature x yb yf]
#wheelcolor  = 'k'
#
#h = []
#
## make wheels
#for front = 1:2
#   for right = [-1 1]
#      h(+1) = rrect(wheel,wheelcolor)' ##ok<AGROW>
#      if front == 2
#         twist(h(),0,0,u(1))
#      
#      twist(h(),right*wheel(4),wheel(4+front))
#   
#
#
## make body
#h(+1) = rrect(body,bodycolor)
#
## make window (hard coded)
#h(+1) = patch([-.8 .8 .7 -.7],.6+.3*[1 1 -1 -1],'w')
#
## headlights
#h(+1) = rrect(headlights(1:3),lightcolor)
#twist(h(),headlights(4),body(2)-headlights(2))
#h(+1) = rrect(headlights(1:3),lightcolor)
#twist(h(),-headlights(4),body(2)-headlights(2))
#
## put rear wheels at (0,0)
#twist(h,0,-wheel(5))
#
## align to x-axis
#twist(h,0,0,-pi/2)
#
## make origin (hard coded)
#ol = 0.1
#ow = 0.01
#h(+1) = patch(ol*[-1 1 1 -1],ow*[1 1 -1 -1],'k')
#h(+1) = patch(ow*[1 1 -1 -1],ol*[-1 1 1 -1],'k')
#
#twist(h,x(1),x(2),x(3))
#
#function twist(obj,x,y,theta)
## a planar twist: rotate object by theta, then translate by (x,y)
#i = 1i
#if nargin == 3
#   theta = 0
#
#for h = obj
#   Z = get(h,'xdata') + i*get(h,'ydata')
#   Z = Z * exp(i*theta)
#   Z = Z + (x + i*y)
#   set(h,'xdata',real(Z),'ydata',imag(Z))
#
#
#function h = rrect(wlc, C)
## draw a rounded rectangle
#if nargin == 1
#   C = 'w'
#
#
#N        = 25
#
#width    = wlc(1)
#length   = wlc(2)
#curve    = wlc(3)
#
#a        = linspace(0,2*pi,4*N)
#z        = curve*exp(1i*a)
#width    = width-curve
#length   = length-curve
#e        = sum( kron(diag(width*[1 -1 -1 1] + 1i*length *[1 1 -1 -1]), ones(1,N)), 1) 
#z        = z+e
#z        = [z z(1)]
#
#h        = patch(real(z),imag(z),C)

# utility functions, singleton-expanded addition and multiplication
def pp(a, b):
    return a+b

def tt(a, b):
    return a*b