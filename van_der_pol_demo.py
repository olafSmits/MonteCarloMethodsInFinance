# -*- coding: utf-8 -*-
"""
@author: chanGimeno and olafSmits
using MATLAB demo code provided by iversity.org.
"""
import numpy as np
from numpy import *
from matplotlib.pylab import *


def eulerIntegrationVanDerPol(t0, x0, p0, mu, T, N):
    """
    Van Der Pol oscillator solved via Euler integration
    INPUT:
        t0 : Initial time
        x0 : Initial position
        p0 : Initial momentum
        mu : damping strength
        T  : Length of integration interval [t0, t0+T]
        N  : Number of time steps

    OUTPUT:
        t  : Times at which the trajectory is monitored
             t(n) = t0 + n Delta T
        x  : Values of the position along the trajectory
        p  : Values of the momentum along the trajectory

    """
    # Size of integration step
    deltaT = T/N

    # initialize monitoring times
    t = np.linspace(t0, t0 + T, N + 1)

    # intialize x and p
    x = np.zeros_like(t)
    p = np.zeros_like(t)

    ## Euler integration
    # initial conditions
    x[0] = x0
    p[0] = p0
    for n in np.arange(N):
        x[n+1] = x[n] + p[n] * deltaT
        p[n+1] = p[n] + (mu * (1 - x[n] * x[n]) * p[n] - x[n]) * deltaT

    return t, x, p


def demo_vanDerPol():
    """
    demo_vanDerPol: numerical integration (Euler) of the Van der Pol oscillator
    """
    # damping parameter
    mu = 2

    # initial conditions
    t0 = 0.
    x0 = 1e-5
    p0 = 1e-5

    # integration parameters
    T = 100.
    N = int(1e5)

    [t,x,p] = eulerIntegrationVanDerPol(t0,x0,p0,mu,T,N)

    # Plot  p(t) as a function of x(t)
    f1 = figure(1); clf()
    h1, = plot(x0,p0,'k',linewidth=1.5)
    h2, = plot(x0,p0,'ro',linewidth=2,markersize=6)
    MIN_X = -2.2;  MAX_X = 2.2
    MIN_Y = -4;    MAX_Y = 4
    xlim(MIN_X, MAX_X)
    ylim(MIN_Y, MAX_Y)
    xlabel('x(t)')
    ylabel('p(t)')
    draw()

    # Plot time series of x(t) and p(t)
    f2 = figure(2); clf()
    subplot(2,1,1)
    h3, = plot(t0,x0)
    xlim(t0, t0+T)
    ylim(MIN_X, MAX_X)
    xlabel('t'); ylabel('x(t)')
    subplot(2,1,2)
    h4, = plot(t0,p0,'m')
    xlim(t0, t0+T)
    ylim(MIN_Y, MAX_Y)
    xlabel('t'); ylabel('p(t)')
    draw()

    pause(3)

    # Update plots
    frameLength = 500
    for i in range(len(t)):
        if(mod(i+1,frameLength) == 0):
            h1.set_data(x[:i], p[:i])
            h2.set_data(x[i], p[i])
            f1.canvas.draw()
            h3.set_data(t[:i], x[:i])
            h4.set_data(t[:i], p[:i])
            pause(0.1)
            f2.canvas.draw()

if __name__ == '__main__':
    demo_vanDerPol()