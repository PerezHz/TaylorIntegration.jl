var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#TaylorIntegration.jl-1",
    "page": "Home",
    "title": "TaylorIntegration.jl",
    "category": "section",
    "text": "ODE integration using Taylor\'s method, and more, in Julia."
},

{
    "location": "index.html#Authors-1",
    "page": "Home",
    "title": "Authors",
    "category": "section",
    "text": "Jorge A. Pérez, Instituto de Ciencias Físicas, Universidad Nacional Autónoma de México (UNAM)\nLuis Benet, Instituto de Ciencias Físicas, Universidad Nacional Autónoma de México (UNAM)"
},

{
    "location": "index.html#License-1",
    "page": "Home",
    "title": "License",
    "category": "section",
    "text": "TaylorIntegration is licensed under the MIT \"Expat\" license."
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "TaylorIntegration.jl is a registered package, and is simply installed by runningpkg> add TaylorIntegration"
},

{
    "location": "index.html#Supporting-and-citing-1",
    "page": "Home",
    "title": "Supporting and citing",
    "category": "section",
    "text": "This package is developed as part of academic research. If you would like to help supporting it, please star the repository as such metrics may help us secure funding. If you use this software, we would be grateful if you could cite our work as follows:J.A. Pérez-Hernández, L. Benet, TaylorIntegration.jl: Taylor Integration in Julia, https://github.com/PerezHz/TaylorIntegration.jl"
},

{
    "location": "index.html#Acknowledgments-1",
    "page": "Home",
    "title": "Acknowledgments",
    "category": "section",
    "text": "We acknowledge financial support from the DGAPA-PAPIIT (UNAM) grant IG-100616. LB acknowledges support through a Cátedra Marcos Moshinsky (2013)."
},

{
    "location": "taylor_method.html#",
    "page": "ODE integration using Taylor\'s method",
    "title": "ODE integration using Taylor\'s method",
    "category": "page",
    "text": ""
},

{
    "location": "taylor_method.html#taylormethod-1",
    "page": "ODE integration using Taylor\'s method",
    "title": "ODE integration using Taylor\'s method",
    "category": "section",
    "text": "Taylor\'s integration method is a quite powerful method to integrate ODEs which are smooth enough, allowing to reach a precision comparable to round-off errors per time-step. A high-order Taylor approximation of the solution (dependent variable) is constructed such that the error is quite small. A time-step is constructed which guarantees the validity of the series; this is used to sum up the Taylor expansion to obtain an approximation of the solution at a later time."
},

{
    "location": "taylor_method.html#rec_rel-1",
    "page": "ODE integration using Taylor\'s method",
    "title": "The recurrence relation",
    "category": "section",
    "text": "Let us consider the followingbeginequation\nlabeleq-ODE\ndotx = f(t x)\nendequationand define the initial value problem with the initial condition x(t_0) = x(0).We write the solution of this equation asbeginequation\nlabeleq-solution\nx = x_0 + x_1 (t-t_0) + x_2 (t-t_0)^2 + cdots +\nx_k (t-t_0)^k + cdots\nendequationwhere the initial condition imposes that x_0 = x(0). Below, we show how to obtain the coefficients x_k of the Taylor expansion of the solution.We assume that the Taylor expansion around t_0 of f(t x(t)) is known, which we write asbeginequation\nlabeleq-rhs\nf(t x(t)) = f_0 + f_1 (t-t_0) + f_2 (t-t_0)^2 + cdots\n+ f_k (t-t_0)^k + cdots\nendequationHere, f_0=f(t_0x_0), and the Taylor coefficients f_k = f_k(t_0) are the k-th normalized derivatives at t_0 given bybeginequation\nlabeleq-normderiv\nf_k = frac1k fracrm d^k f rm d t^k(t_0)\nendequationThen, we are assuming that we know how to obtain f_k; these coefficients are obtained using TaylorSeries.jl.Substituting Eq. (\\ref{eq-solution}) in (\\ref{eq-ODE}), and equating powers of t-t_0, we obtainbeginequation\nlabeleq-recursion\nx_k+1 = fracf_k(t_0)k+1 quad k=01dots\nendequationTherefore, the coefficients of the Taylor expansion (\\ref{eq-solution}) are obtained recursively using Eq. (\\ref{eq-recursion})."
},

{
    "location": "taylor_method.html#time-step-1",
    "page": "ODE integration using Taylor\'s method",
    "title": "Time step",
    "category": "section",
    "text": "In the computer, the expansion (\\ref{eq-solution}) has to be computed to a finite order. We shall denote by K the order of the series. Clearly, the larger the order K, the more accurate the obtained solution is.The theorem of existence and uniqueness of the solution of Eq.~(\\ref{eq-ODE}) ensures that the Taylor expansion converges. Then, assuming that K is large enough to be within the convergent tail. We introduce the parameter epsilon_textrmtol  0 to control how large is the last term. The idea is to set this parameter to a small value, usually smaller than the machine-epsilon. Denoting by h = t_1-t_0 the time step, then  x_K  h^K le epsilon_textrmtol, we obtainbeginequation\nlabeleq-h\nh le Big(fracepsilon_textrmtol x_K Big)^1K\nendequationEquation (\\ref{eq-h}) represents the maximum time-step which is consistent with epsilon_textrmtol, K and the assumption of being within the convergence tail. Notice that the arguments exposed above simply ensure that h is a maximum time-step, but any other smaller than h can be used since the series is convergent in the open interval tin(t_0-ht_0+h).Finally, from Eq. (\\ref{eq-solution}) with (\\ref{eq-h}) we obtain x(t_1) = x(t_0+h), which is again an initial value problem."
},

{
    "location": "lyapunov_spectrum.html#",
    "page": "Lyapunov spectrum",
    "title": "Lyapunov spectrum",
    "category": "page",
    "text": ""
},

{
    "location": "lyapunov_spectrum.html#lyap-1",
    "page": "Lyapunov spectrum",
    "title": "Lyapunov spectrum",
    "category": "section",
    "text": "Here we describe the background of the Lyapunov spectra computations in TaylorIntegration.jl. Our implementation follows the numerical method of Benettin et al. [1], [2], which itself is based on Oseledet\'s multiplicative ergodic theorem [3]. Namely, simultaneously to the integration of the equations of motion, we integrate the 1st-order variational equations associated to them.In general, given a dynamical system defined by the equations of motionbeginequation\nlabeleq-ODE-l\ndotx = f(t x)\nendequationalong with the initial condition x(t_0) = x_0, then the first-order variational equations associated to this system arebeginequation\nlabelvar-eqs\ndotxi = (operatornameDf)(x(t))cdot xi\nendequationwhere (operatornameDf)(x(t)) is the Jacobian of the function f with respect to the dependent variable x, evaluated at time t, for a given solution x(t) to the equations of motion. The variable xi denotes a matrix, whose initial condition is xi(t_0) = mathbb1_n, the ntimes n identity matrix, where n is the degrees of freedom or number of dependent variables x.For the actual computation of the Lyapunov spectrum, we proceed as follows. During the simultaneous numerical integration of the equations of motion and the variational equations, at fixed time intervals t_k = kcdot Delta t, k = 1 2 ldots we perform a QR decomposition over xi(t_k), the solution of the variational equations at time t_k. That is, we factorize xi(t_k) as xi(t_k)=Q_kcdot R_k, where Q_k is an orthogonal ntimes n matrix and R_k is an upper triangular ntimes n matrix with positive diagonal elements. The diagonal elements R_iik are the growth factors from which the l-th Lyapunov exponent is computed at time t_kbeginequation\nlabellyap-spec\nlambda_l = sum_m=1^k fraclog (R_llm)kcdot Delta t\nendequationIn turn, the matrix Q is substituted into xi(t_k) as the new (scaled) initial condition.The equations of motion together with the variational equations are integrated up to time t_k+1 using Taylor\'s method. We note that each time step of the integration is determined using the normalized derivatives of x and the tolerance epsilon_textrmtol. This process is repeated until a prescribed t_textrmmax is reached.This example illustrates the computation of the Lyapunov spectrum for the Lorenz system."
},

{
    "location": "lyapunov_spectrum.html#refsL-1",
    "page": "Lyapunov spectrum",
    "title": "References",
    "category": "section",
    "text": "[1] Benettin G., Galgani L., Giorgilli A., Strelcyn J.M., 1980, Meccanica, 15, 9[2] Benettin G., Galgani L., Giorgilli A., Strelcyn J.M., 1980, Meccanica, 15, 21[3] Oseledets V. I., 1968, Trudy Moskovskogo Matematicheskogo Obshchestva, 19, 179"
},

{
    "location": "jet_transport.html#",
    "page": "Jet transport",
    "title": "Jet transport",
    "category": "page",
    "text": ""
},

{
    "location": "jet_transport.html#jettransport-1",
    "page": "Jet transport",
    "title": "Jet transport",
    "category": "section",
    "text": "In this section we describe the jet transport capabilities included in TaylorIntegration.jl. Jet transport is a tool that allows the propagation under the flow of a small neighborhood in phase space around a given initial condition, instead of propagating a single initial condition only.To compute the propagation of mathbfx_0 + delta mathbfx, where delta mathbfx are independent small displacements in phase space around the initial condition mathbfx_0, one has to solve high-order variational equations. The idea is to treat mathbfx_0 + delta mathbfx as a truncated polynomial in the delta mathbfx variables. The maximum order of this polynomial has to be fixed in advance.Jet transport works in general with any ordinary ODE solver, provided the chosen solver supports computations using multi-variate polynomial algebra."
},

{
    "location": "jet_transport.html#A-simple-example-1",
    "page": "Jet transport",
    "title": "A simple example",
    "category": "section",
    "text": "Following D. Pérez-Palau et al [1], let us consider the differential equations for the harmonic oscillator:begineqnarray*\ndotx  =  y \ndoty  =  -x\nendeqnarray*with the initial condition mathbfx_0=x_0 y_0^T. We illustrate jet transport techniques using Euler\'s methodbeginequation*\nmathbfx_n+1 = mathbfx_n + h mathbff(mathbfx_n)\nendequation*Instead of considering the initial conditions mathbfx_0, we consider the time evolution of the polynomialbeginequation*\nP_0mathbfx_0(deltamathbfx) = x_0+delta x y_0 + delta y^T\nendequation*where delta x and delta y are small displacements. Below we concentrate in polynomials of order 1 in delta x and delta y; since the equations of motion of the harmonic oscillator are linear, there are no higher order terms.Using Euler\'s method we obtainbegineqnarray*\n  mathbfx_1  = \n  left(\n    beginarrayc\n    x_0 + h y_0 \n    y_0 - h x_0\n    endarray\n  right)\n  + left(\n      beginarraycc\n         1  h \n        -h  1\n      endarray\n    right)\n    left(\n      beginarrayc\n        delta x\n        delta y\n      endarray\n    right) \n  mathbfx_2  = \n  left(\n    beginarrayc\n    1-h^2 x_0 + 2 h y_0 \n    1-h^2 y_0 - 2 h x_0\n    endarray\n  right)\n  + left(\n    beginarraycc\n      1-h^2  2 h \n      -2 h  1-h^2\n    endarray\n    right)\n    left(\n      beginarrayc\n        delta x\n        delta y\n      endarray\n    right)\nendeqnarray*The first terms in the expressions for mathbfx_1 and mathbfx_2 above correspond to the result of an Euler integration step using the initial conditions only. The other terms are the (linear) corrections which involve the small displacements delta x and delta y.In general, for differential equations involving non-linear terms, the resulting expansions in delta x and delta y will reflect aspects of the non-linearities of the ODEs. Clearly, jet transport techniques allow to address stability properties beyond the linear case, though memory constraints may play a role. See this example illustrating the implementation for the simple pendulum, and this one illustrating the construction of a Poincaré map with Jet transport techniques."
},

{
    "location": "jet_transport.html#refsJT-1",
    "page": "Jet transport",
    "title": "References",
    "category": "section",
    "text": "[1] D. Pérez-Palau, Josep J. Masdemont, Gerard Gómez, 2015, Celest. Mech. Dyn. Astron. 123, 239."
},

{
    "location": "simple_example.html#",
    "page": "Infinity in finite time",
    "title": "Infinity in finite time",
    "category": "page",
    "text": ""
},

{
    "location": "simple_example.html#example1-1",
    "page": "Infinity in finite time",
    "title": "Infinity in finite time",
    "category": "section",
    "text": ""
},

{
    "location": "simple_example.html#Illustration-of-the-method-1",
    "page": "Infinity in finite time",
    "title": "Illustration of the method",
    "category": "section",
    "text": "We shall illustrate first with a simple example how the method explicitly constructs the solution, and how to use the package to obtain it.We consider the differential equation given bybeginequation\nlabeleq-example1\ndotx = x^2\nendequationwith the initial condition x(0)=x_0, whose exact solution readsbeginequation\nlabeleq-sol1\nx(t) = fracx_01-x_0t\nendequationWe shall implement the construction of this example explicitly, which illustrates the way TaylorIntegration.jl is conceived.The initial condition defines the 0-th order approximation, i.e., x(t) = x_0 + mathcalO(t^1). We now write the solution as x(t) = x_0 + x_1t + mathcalO(t^2), and we want to determine x_1. Substituting this solution into the RHS of (\\ref{eq-example1}), yieldsx^2 = x_0^2 + 2 x_0 x_1 t + x_1^2 t^2 =\n x_0^2 + mathcalO(t^1)where in the last equality we have kept all terms up to order 0, since we want to determine x_1, and the recursion formula requires for that the 0-th order term of the Taylor expansion of the RHS of the equation of motion. Hence, we have f_0=x_0^2, which using the recursion relation x_k+1 = f_k(k+1) yields x_1 = x_0^2.Following the same procedure, we write x(t) = x_0 + x_1 t + x_2 t^2 + mathcalO(t^3), andx^2 = x_0^2 + 2 x_0 x_1 t + mathcalO(t^2)where we kept all terms up to order 1. We thus have f_1=2 x_0 x_1 = 2 x_0^3, which then yields x_2 = x_0^3. Repeating this calculation, we obtainbeginequation\nlabeleq-solTaylor\nx(t) = x_0 + x_0^2 t + x_0^3 t^2 + cdots + x_0^k+1 t^k + cdots\nendequationThe solution given by Eq. (\\ref{eq-solTaylor}) is a geometrical series, which is identical to the exact solution, Eq. (\\ref{eq-sol1}). Yet, it is not obvious from the solution that it is only defined for t1x_0. To see this, we obtain the step size, as described previously, for the series truncated to order k. The Taylor coefficient of order k is x_k=x_0^k+1, so the time step ish  Big(fracepsilon_textrmtolx_0^k+1Big)^1k =\nfracepsilon_textrmtol^1kx_0^1+1kIn the limit ktoinfty we obtain h  h_textrmmax=1x_0, which is the domain of existence of the exact solution.Below, we shall fix a maximum order for the expansion. This entails a truncation error which is somewhat controlled through the absolute tolerance epsilon_textrmtol. The key to a correct use of Taylor\'s method is to impose a quite small value of epsilon_textrmtol together with a large enough order of the expansion."
},

{
    "location": "simple_example.html#Implementation-1",
    "page": "Infinity in finite time",
    "title": "Implementation",
    "category": "section",
    "text": "We shall illustrate how to use TaylorIntegration.jl to integrate Eq. (\\ref{eq-example1}) for the initial condition x(0)=3. Notice that according to the exact solution Eq. (\\ref{eq-sol1}), the solution only exists for tt_mathrmmax =13; in addition, we note that this number can not be represented exactly as a floating-point.We first load the required packages and define a function which represents the equation of motion.using TaylorIntegration, Plots\ndiffeq(t, x) = x^2;note: Note\nIn TaylorIntegration.jl, the convention for writing the function representing the equations of motion is to use first the independent variable (t), followed by the dependent variables (x) and then the derivatives defining the equations of motion (dx). For a single ODE, as in the present case, we omit the last argument which is returned, and avoid using vectors; for more ODEs, both x and dx are preallocated vectors and the function mutates (modifies) dx.Now, we integrate the equations of motion using taylorinteg; despite of the fact that the solution only exists for tt_textrmmax, below we shall try to compute it up to t_textrmend=034; as we shall see, Taylor\'s method takes care of this. For the integration presented below, we use a 28-th series expansion, with epsilon_textrmtol = 10^-20, and compute up to 150 integration steps.tT, xT = taylorinteg(diffeq, 3.0, 0.0, 0.34, 28, 1e-20, maxsteps=150);We first note that the last point of the calculation does not exceed t_textrmmax.tT[end]Increasing the maxsteps parameter pushes tT[end] closer to t_textrmmax but it actually does not reach this value.Figure 1 displays the computed solution as a function of time, in log scale.plot(tT, log10.(xT), shape=:circle)\nxlabel!(\"t\")\nylabel!(\"log10(x(t))\")\nxlims!(0,0.34)\ntitle!(\"Fig. 1\")Clearly, the solution diverges without bound when tto t_textrmmax = 13, i.e., x(t) approaches infinity in finite time.Figure 2 shows the relative difference between the numerical and the analytical solution in terms of time.exactsol(t, x0) = x0 / (1 - x0 * t)\nδxT = abs.(xT .- exactsol.(tT, 3.0)) ./ exactsol.(tT, 3.0);\nplot(tT[6:end], log10.(δxT[6:end]), shape=:circle)\nxlabel!(\"t\")\nylabel!(\"log10(dx(t))\")\nxlims!(0, 0.4)\ntitle!(\"Fig. 2\")To put in perspective how good is the constructed solution, we impose (arbitrarily) a relative accuracy of 10^-13; the time until such accuracy is satisfied is given by:indx = findfirst(δxT .> 1.0e-13);\nesol = exactsol(tT[indx-1],3.0);\ntT[indx-1], esol, eps(esol)Note that, the accuracy imposed in terms of the actual value of the exact solution means that the difference of the computed and the exact solution is essentially due to the eps of the computed value."
},

{
    "location": "kepler.html#",
    "page": "The Kepler problem",
    "title": "The Kepler problem",
    "category": "page",
    "text": ""
},

{
    "location": "kepler.html#kepler_problem-1",
    "page": "The Kepler problem",
    "title": "The Kepler problem",
    "category": "section",
    "text": "The Kepler problem corresponds to the study of the motion of two bodies which are influenced by their mutual gravitational attraction. In the center of mass and relative coordinates, the problem is reduced to the motion of one body of mass m = m_1 m_2  M, which we shall refer as particle below, attracted gravitationally by another located at rest at the origin of mass M=m_1+m_2.In cartesian coordinates, the equations of motion can be written asbegineqnarray*\ndotx = v_x\ndoty = v_y\ndotv_x = - fracG M x(x^2 + y^2)^32\ndotv_y = - fracG M y(x^2 + y^2)^32\nendeqnarray*For concreteness, we fix mu = G M = 1. The coordinates x and y are the relative coordinates (to the center of mass) of the particle, and v_x and v_y its velocity. The function kepler_eqs! mutates the vectors corresponding to the LHS of the equations of motion.function kepler_eqs!(t, q, dq)\n    dq[1] = q[3]\n    dq[2] = q[4]\n    rr = ( q[1]^2 + q[2]^2 )^(3/2)\n    dq[3] = - q[1] / rr\n    dq[4] = - q[2] / rr\nend;For suitable initial conditions (such that the total energy is negative) the solutions are ellipses with one focus at the origin, which can be parameterized in terms of its semi-major axis a and its eccentricity e. We set the initial conditions for the particle at periapse, which we locate on the positive x-axis. Using the semimajor axis and the eccentricity, we write them asbegineqnarray*\nx_0  =  a (1-e)\ny_0  =  0\nv_x_0  =  0\nv_y_0  =  fracl_zx_0 = m fracsqrtmu a (1-e^2)x_0\nendeqnarray*where l_z is the angular momentum. We set the mass of the particle m=1, the semi-major axis a=1 and the eccentricity e=08. Kepler\'s third law defines the period of the motion as T= 2pi a^32.const mu = 1.0\nconst mass = 1.0\nconst aKep = 1.0\nconst eKep = 0.8;The initial conditions are then set using ini_condfunction ini_cond(a, e)\n    x0  = a*(one(e)-e)\n    vy0 = mass * sqrt( mu * a * (1-e^2) ) / x0\n    y0  = zero(vy0)\n    vx0 = zero(vy0)\n    return [x0, y0, vx0, vy0]\nend\nq0 = ini_cond(aKep, eKep)We now perform the integration, using a 28 order expansion and absolute tolerance of 10^-20.using TaylorIntegration, Plots\nt, q = taylorinteg(kepler_eqs!, q0, 0.0, 10000*2pi, 28, 1.0e-20, maxsteps=700000);\nt[end], q[end,:]We first plot the orbit.x = view(q, :, 1)\ny = view(q, :, 2)\nvx = view(q, :, 3)\nvy = view(q, :, 4)\nplot(x, y, legend=false)\nscatter!([0], [0], shape=:circle, ms=5)\nxaxis!(\"x\", (-2.0, 0.5))\nyaxis!(\"y\", (-1.0, 1.0))\ntitle!(\"Fig. 1\")The following functions allow us to calculate the energy and angular momentum using cartesian coordinates.function energy( x, y, vx, vy )\n    kinetic = 0.5 * (vx*vx + vy*vy)\n    r = sqrt( x*x + y*y)\n    potential = - mu * mass / r\n    return kinetic + potential\nend\nlz( x, y, vx, vy ) = mass * ( x*vy - y*vx ) ;We use the change in energy and angular momentum of the orbit with respect to the initial value of the corresponding quantity as a function of time. These quantities are expressed in units of the local epsilon of the initial energy or angular momentum, respectively. This serves to illustrate the accuracy of the calculation, shown in Figure 2 and 3.e0 = energy(q0...)\nδE = (energy.(x,y,vx,vy) .- e0) ./ eps(e0)\nplot(t, δE)\nxlabel!(\"t\")\nylabel!(\"dE\")\ntitle!(\"Fig. 2\")lz0 = lz(q0...)\nδlz = (lz.(x,y,vx,vy) .- lz0) ./ eps(lz0)\nplot(t, δlz)\nxlabel!(\"t\")\nylabel!(\"dlz\")\ntitle!(\"Fig. 3\")These errors are reminiscent of random walks.The maximum absolute errors of the energy and angular momentum aremaximum( abs.(energy.(x,y,vx,vy) .- e0) ), maximum( abs.(lz.(x,y,vx,vy) .- lz0) )"
},

{
    "location": "lorenz_lyapunov.html#",
    "page": "Lyapunov spectrum of Lorenz system",
    "title": "Lyapunov spectrum of Lorenz system",
    "category": "page",
    "text": ""
},

{
    "location": "lorenz_lyapunov.html#lyap_lorenz-1",
    "page": "Lyapunov spectrum of Lorenz system",
    "title": "Lyapunov spectrum of Lorenz system",
    "category": "section",
    "text": "Here, we present the calculation of the Lyapunov spectrum of the Lorenz system, using TaylorIntegration.jl. The computation involves evaluating the 1st order variational equations dot xi = J cdot xi for this system, where J = operatornameDf is the Jacobian. By default, the numerical value of the Jacobian is computed using automatic differentiation techniques implemented in TaylorSeries.jl, which saves us from writing down explicitly the Jacobian. Conversely, this can be used to check a function implementing the Jacobian. As an alternative, specially important if performance is critical, the user may provide a Jacobian function.The Lorenz system is the ODE defined as:begineqnarray*\n    dotx_1  =  sigma(x_2-x_1) \n    dotx_2  =  x_1(rho-x_3)-x_2 \n    dotx_3  =  x_1x_2-beta x_3\nendeqnarray*where sigma, rho and beta are constant parameters.First, we write a Julia function which evaluates (in-place) the Lorenz system:#Lorenz system ODE:\nfunction lorenz!(t, x, dx)\n    dx[1] = σ*(x[2]-x[1])\n    dx[2] = x[1]*(ρ-x[3])-x[2]\n    dx[3] = x[1]*x[2]-β*x[3]\n    nothing\nend\nnothing #hideBelow, we use the the parameters sigma = 160, beta = 4 and rho = 4592.#Lorenz system parameters\n#we use the `const` prefix in order to help the compiler speed things up\nconst σ = 16.0\nconst β = 4.0\nconst ρ = 45.92\nnothing # hideWe define the initial conditions, the initial and final integration times for the integration:const x0 = [19.0, 20.0, 50.0] #the initial condition\nconst t0 = 0.0     #the initial time\nconst tmax = 100.0 #final time of integration\nnothing # hideSince the diagonal of the Jacobian is constant, the sum of the Lyapunov spectrum has to be equal to that value. We calculate this trace using TaylorSeries.jl, and after the numerical integration, we will come back to check if this value is conserved (or approximately conserved) as a function of time.# Note that TaylorSeries.jl is @reexport-ed by TaylorIntegration.jl\n# Calculate trace of Lorenz system Jacobian via TaylorSeries.jacobian:\nimport LinearAlgebra: tr\nusing TaylorIntegration\nxi = set_variables(\"δ\", order=1, numvars=length(x0))\nx0TN = x0 .+ xi\ndx0TN = similar(x0TN)\nlorenz!(t0, x0TN, dx0TN)\njjac = jacobian(dx0TN)\nlorenztr = tr(jjac) #trace of Lorenz system Jacobian matrix\nnothing # hideAs explained above, the user may provide a function which computes the Jacobian of the ODE in-place:#Lorenz system Jacobian (in-place):\nfunction lorenz_jac!(jac, t, x)\n    jac[1,1] = -σ + zero(x[1])\n    jac[1,2] = σ + zero(x[1])\n    jac[1,3] = zero(x[1])\n    jac[2,1] = ρ - x[3]\n    jac[2,2] = -1.0 + zero(x[1])\n    jac[2,3] = -x[1]\n    jac[3,1] = x[2]\n    jac[3,2] = x[1]\n    jac[3,3] = -β + zero(x[1])\n    nothing\nend\nnothing # hidenote: Note\nWe use of zero(x[1]) in the function lorenz_jac! when the RHS consists of a numeric value; this is needed to allow the proper promotion of the variables to carry out Taylor\'s method.We can actually check the consistency of lorenz_jac! with the computation of the jacobian using automatic differentiation techniques. Below we use the initial conditions x0, but it is easy to generalize this.lorenz_jac!(jjac, t0, x0)  # update the matrix `jjac` using Jacobian provided by the user\njacobian(dx0TN) == jjac    # `dx0TN` is obtained via automatic differentiationNow, we are ready to perform the integration using lyap_taylorinteg function, which integrates the 1st variational equations and uses Oseledets\' theorem. The expansion order will be 28 and the local absolute tolerance will be 10^-20. lyap_taylorinteg will return three arrays: one with the evaluation times, one with the values of the dependent variables (at the time of evaluation), and another one with the values of the Lyapunov spectrum.We first carry out the integration computing internally the Jacobiantv, xv, λv = lyap_taylorinteg(lorenz!, x0, t0, tmax, 28, 1e-20; maxsteps=2000000);\nnothing # hideNow, the integration is obtained exploiting lorenz_jac!:tv_, xv_, λv_ = lyap_taylorinteg(lorenz!, x0, t0, tmax, 28, 1e-20, lorenz_jac!; maxsteps=2000000);\nnothing # hideIn terms of performance the second method is about ~50% faster than the first.We check the consistency of the orbits computed by the two methods:tv == tv_, xv == xv_, λv == λv_As mentioned above, a more subtle check is related to the fact that the trace of the Jacobian is constant in time, which must coincide with the sum of all Lyapunov exponents. Using its initial value lorenztr, we compare it with the final Lyapunov exponents of the computation, and obtainsum(λv[end,:]) ≈ lorenztr, sum(λv_[end,:]) ≈ lorenztr, sum(λv[end,:]) ≈ sum(λv_[end,:])Above we checked the approximate equality; we now show that the relative error is quite small and comparable with the local machine epsilon value around lorenztr:abs(sum(λv[end,:])/lorenztr - 1), abs(sum(λv_[end,:])/lorenztr - 1), eps(lorenztr)Therefore, the numerical error is dominated by roundoff errors in the floating point arithmetic of the integration. We will now proceed to plot our results. First, we plot Lorenz attractor in phase spaceusing Plots\nplot(xv[:,1], xv[:,2], xv[:,3], leg=false)We display now the Lyapunov exponents as a function of time:using Plots\nnothing # hide\nplot(tv, λv[:,1], label=\"L_1\", legend=:right)\nplot!(tv, λv[:,2], label=\"L_2\")\nplot!(tv, λv[:,3], label=\"L_3\")\nxlabel!(\"time\")\nylabel!(\"L_i, i=1,2,3\")\ntitle!(\"Lyapunov exponents vs time\")This plot shows that the calculation of the Lyapunov exponents has converged."
},

{
    "location": "pendulum.html#",
    "page": "Jet transport: the simple pendulum",
    "title": "Jet transport: the simple pendulum",
    "category": "page",
    "text": ""
},

{
    "location": "pendulum.html#pendulum-1",
    "page": "Jet transport: the simple pendulum",
    "title": "Jet transport: the simple pendulum",
    "category": "section",
    "text": "In this example we illustrate the use of jet transport techniques in TaylorIntegration.jl for the simple pendulum. We propagate a neighborhood U_0 around an initial condition q_0 parametrized by the sum q_0+xi, where q_0=(x_0p_0) represents the coordinates of the initial condition in phase space, and xi=(xi_1xi_2) represents an small variation with respect to this initial condition. We re-interpret each component of the sum q_0+xi as a multivariate polynomial in the variables xi_1 and xi_2; below, the maximum order of the multivariate polynomial is fixed at 8. We propagate these multivariate polynomials in time using Taylor\'s method.The simple pendulum is defined by the Hamiltonian H(x p) = frac12p^2-cos x; the corresponding equations of motion are given bybegineqnarray*\ndotx = p \ndotp = -sin x\nendeqnarray*We integrate this problem for a neighborhood U_0 around the initial condition q_0 = (x(t_0) p(t_0)) = (x_0 p_0). For concreteness we take p_0=0 and choose x_0 such that the pendulum librates; that is, we will choose a numerical value for the energy E=H(x_0p_0)=-cos x_0 such that the pendulum\'s motion in phase space is \"below\" (inside) the region bounded by the separatrix. In this case, the libration period T of the pendulum isbeginequation*\nT=frac4sqrt2int_0^x_0fracdxsqrtcos x_0-cos x\nendequation*which can be expressed in terms of the complete elliptic integral of the first kind, K:beginequation*\nT=4K(sin(x_02))\nendequation*The Hamiltonian for the simple pendulum is:H(x) = 0.5x[2]^2-cos(x[1])\nnothing # hideThe equations of motion are:function pendulum!(t, x, dx)\n    dx[1] = x[2]\n    dx[2] = -sin(x[1])\nend\nnothing # hideWe define the TaylorN variables necessary to perform the jet transport; varorder represents the maximum order of expansion in the variations xi.const varorder = 8\nusing TaylorIntegration\nξ = set_variables(\"ξ\", numvars=2, order=varorder)Note that TaylorSeries.jl is @reexport-ed internally by TaylorIntegration.jl.The nominal initial condition is:q0 = [1.3, 0.0]The corresponding initial value of the energy is:H0 = H(q0)The parametrization of the neighborhood U_0 is represented byq0TN = q0 .+ ξTo understand how the jet transport technique works, we shall evaluate the Hamiltonian at q_0+xi in order to obtain the 8-th order Taylor expansion of the Hamiltonian with respect to the variations xi, around the initial condition q_0:H(q0TN)Note that the 0-th order term of the expression above is equal to the value H(q0), as expected.Below, we set some parameters for the Taylor integration. We use a method of taylorinteg which returns the solution at t0, t0+integstep, t0+2integstep,...,tmax, where t0 and tmax are the initial and final times of integration, whereas integstep is a time interval chosen by the user; we use the variable tv = t0:integstep:tmax for this purpose and choose integstep as fracT8.order = 28     #the order of the Taylor expansion wrt time\nabstol = 1e-20 #the absolute tolerance of the integration\nusing Elliptic # we use Elliptic.jl to evaluate the elliptic integral K\nT = 4*Elliptic.K(sin(q0[1]/2)^2) #the libration period\nt0 = 0.0        #the initial time\ntmax = 6T       #the final time\nintegstep = T/8 #the time interval between successive evaluations of the solution vector\nnothing # hideWe perform the Taylor integration using the initial condition x0TN, during 6 periods of the pendulum (i.e., 6T), exploiting multiple dispatch:tv = t0:integstep:tmax # the times at which the solution will be evaluated\nxv = taylorinteg(pendulum!, q0TN, tv, order, abstol)\nnothing # hideThe integration above works for any initial neighborhood U_0 around the nominal initial condition q_0, provided it is sufficiently small.We will consider the particular case where U_0 is a disk of radius r = 005, centered at q_0; that is U_0= q_0+xixi=(rcosphirsinphi) phiin02pi)  for a given radius r0. We will denote by U_t the propagation of the initial neighborhood U_0 evaluated at time t. Also, we denote by q(t) the coordinates of the nominal solution at time t: q(t)=(x(t)p(t)). Likewise, we will denote the propagation at time t of a given initial variation xi_0 by xi(t). Then, we can compute the propagation of the boundary partial U_t of the neighborhood U_t.polar2cart(r, ϕ) = [r*cos(ϕ), r*sin(ϕ)] # convert radius r and angle ϕ to cartesian coordinates\nr = 0.05 #the radius of the neighborhood\nϕ = 0.0:0.1:(2π+0.1) #the values of the angle\nξv = polar2cart.(r, ϕ)\nnothing # hideWe evaluate the jet at partial U_x(t) (the boundary of U_x(t)) at each value of the solution vector xv; we organize these values such that we can plot them later:xjet_plot = map(λ->λ.(ξv), xv[:,1])\npjet_plot = map(λ->λ.(ξv), xv[:,2])\nnothing # hideAbove, we have exploited the fact that Array{TaylorN{Float64}} variables are callable objects. Now, we evaluate the jet at the nominal solution, which corresponds to xi=(00), at each value of the solution vector xv:x_nom = xv[:,1]()\np_nom = xv[:,2]()\nnothing # hideFinally, we shall plot the nominal solution (black dots), as well as the evolution of the neighborhood U_0 (in colors), each frac18th of a period T. The initial condition corresponds to the black dot situated at q_0=(130)using Plots\nplot( xjet_plot, pjet_plot,\n    xaxis=(\"x\",), yaxis=(\"p\",),\n    title=\"Simple pendulum phase space\",\n    leg=false, aspect_ratio=1.0\n)\nscatter!( x_nom, p_nom,\n    color=:black,\n    m=(1,2.8,stroke(0))\n)"
},

{
    "location": "root_finding.html#",
    "page": "Poincaré maps",
    "title": "Poincaré maps",
    "category": "page",
    "text": ""
},

{
    "location": "root_finding.html#rootfinding-1",
    "page": "Poincaré maps",
    "title": "Poincaré maps",
    "category": "section",
    "text": "In this example, we shall illustrate how to construct a Poincaré map associated with the surface of section y=0, dot y0, for E=01025 for the Hénon-Heiles system. This is equivalent to find the roots of an appropriate function g(t, x, dx). We illustrate the implementation using many initial conditions (Monte Carlo like implementation), and then compare the results with the use of jet transport techniques."
},

{
    "location": "root_finding.html#Monte-Carlo-simulation-1",
    "page": "Poincaré maps",
    "title": "Monte Carlo simulation",
    "category": "section",
    "text": "The Hénon-Heiles system is a 2-dof Hamiltonian system used to model the (planar) motion of a star around a galactic center. The Hamiltonian is given by H = (p_x^2+p_y^2)2 + (x^2+y^2)2 + lambda (x^2y-y^33), from which the equations of motion can be obtained; below we concentrate in the case lambda=1.# Hamiltonian\nV(x,y) = 0.5*( x^2 + y^2 )+( x^2*y - y^3/3)\nH(x,y,p,q) = 0.5*(p^2+q^2) + V(x, y)\nH(x) = H(x...)\n\n# Equations of motion\nfunction henonheiles!(t, x, dx)\n    dx[1] = x[3]\n    dx[2] = x[4]\n    dx[3] = -x[1]-2x[2]*x[1]\n    dx[4] = -x[2]-(x[1]^2-x[2]^2)\n    nothing\nend\nnothing # hideWe set the initial energy, which is a conserved quantity; x0 corresponds to the initial condition, which will be properly adjusted to be in the correct energy surface.# initial energy and initial condition\nconst E0 = 0.1025\nx0 = [0.0, 0.45335, 0.0, 0.0]\nnothing # hideIn order to be able to generate (random) initial conditions with the appropriate energy, we write a function px, which depends on x, y, py and the energy E that returns the value of px>0 for which the initial condition [x, y, px, py] has energy E:# px: select px0>0 such that E=E0\npx(x, E) = sqrt(2(E-V(x[1], x[2]))-x[4]^2)\n\n# px!: in-place version of px; returns the initial condition\nfunction px!(x, E)\n    mypx = px(x, E)\n    x[3] = mypx\n    return x\nend\n\n# run px!\npx!(x0, E0)Let\'s check that the initial condition x0 has actually energy equal to E0, up to roundoff accuracy:H(x0)The scalar function g, which may depend on the time t, the vector of dependent variables x and even the velocities dx, defines the surface of section by means of the condition g(t, x, dx) == 0; g should return a variable of type eltype(x). In the present case, it is defined as# x=0, px>0 section\nfunction g(t, x, dx)\n    px_ = constant_term(x[3])\n    # if px > 0...\n    if px_ > zero(px_)\n        # ...return x\n        return x[1]\n    else\n        #otherwise, discard the crossing\n        return zero(x[1])\n    end\nend\nnothing # hidenote: Note\nNote that in the definition of g we want to make sure that we only take the \"positive\" crossings through the surface of section x=0; hence the if...else... block.We initialize some auxiliary arrays, where we shall save the solutions:# number of initial conditions\nnconds = 100\ntvSv = Vector{Vector{Float64}}(undef, nconds)\nxvSv = Vector{Matrix{Float64}}(undef, nconds)\ngvSv = Vector{Vector{Float64}}(undef, nconds)\nx_ini = similar(x0)\nnothing # hideWe generate nconds random initial conditions in a small neighborhood around x0 and integrate the equations of motion from t0=0 to tmax=135, using a polynomial of order 25 and absolute tolerance 1e-25:using TaylorIntegration\n\nfor i in 1:nconds\n    rand1 = rand()\n    rand2 = rand()\n    x_ini .= x0 .+ 0.005 .* [0.0, sqrt(rand1)*cos(2pi*rand2), 0.0, sqrt(rand1)*sin(2pi*rand2)]\n    px!(x_ini, E0)\n\n    tv_i, xv_i, tvS_i, xvS_i, gvS_i = taylorinteg(henonheiles!, g, x_ini, 0.0, 135.0,\n        25, 1e-25, maxsteps=30000);\n    tvSv[i] = vcat(0.0, tvS_i)\n    xvSv[i] = vcat(transpose(x_ini), xvS_i)\n    gvSv[i] = vcat(0.0, gvS_i)\nend\nnothing # hideWe generate an animation with the solutionsusing Plots\npoincare_anim1 = @animate for i=1:21\n    scatter(map(x->x[i,2], xvSv), map(x->x[i,4], xvSv), label=\"$(i-1)-th iterate\",\n        m=(1,stroke(0)), ratio=:equal)\n    xlims!(0.08, 0.48)\n    ylims!(-0.13, 0.13)\n    xlabel!(\"y\")\n    ylabel!(\"py\")\n    title!(\"Hénon-Heiles Poincaré map (21 iterates)\")\nend\ngif(poincare_anim1, \"poincareanim1.gif\", fps = 2);\nnothing # hide(Image: Poincaré map for the Hénon Heiles system)"
},

{
    "location": "root_finding.html#jettransport2-1",
    "page": "Poincaré maps",
    "title": "Jet transport",
    "category": "section",
    "text": "Now, we illustrate the use of jet transport techniques in the same example, that is, we propagate a neighborhood around x0, which will be plotted in the Poincaré map. We first define the vector of small increments of the phase space variables, xTN; we fix the maximum order of the polynomial expansion in these variables to be 4. Then, x0TN is the neighborhood in the 4-dimensional phase space around x0.xTN = set_variables(\"δx δy δpx δpy\", numvars=length(x0), order=4)\nx0TN = x0 .+ xTN\nnothing # hideAs it was shown above, x0 belongs to the energy surface H(x0) = E_0 = 01025; yet, as it was defined above, the set of phase space points denoted by x0TN includes points that belong to other energy surfaces. This can be noticed by computing H(x0TN)H(x0TN)Clearly, the expression above may contain points whose energy is different from E0. As it was done above, we shall fix the px component of x0TN so all points of the neighborhood are in the same energy surface.px!(x0TN, E0) # Impose that all variations are on the proper energy shell!\nH(x0TN)We notice that the coefficients of all monomials whose order is not zero are very small, and the constant_term is E0.In order to properly handle this case, we need to extend the definition of g to be useful for Taylor1{TaylorN{T}} vectors.#specialized method of g for Taylor1{TaylorN{T}}\'s\nfunction g(t, x::Array{Taylor1{TaylorN{T}},1}, dx::Array{Taylor1{TaylorN{T}},1}) where {T<:Number}\n    px_ = constant_term(constant_term(x[3]))\n    if px_ > zero( T )\n        return x[1]\n    else\n        return zero(x[1])\n    end\nend\nnothing # hideWe are now set to carry out the integration.tvTN, xvTN, tvSTN, xvSTN, gvSTN = taylorinteg(henonheiles!, g, x0TN, 0.0, 135.0, 25, 1e-25, maxsteps=30000);\nnothing # hideWe define some auxiliary arrays, and then make an animation with the results for plotting.#some auxiliaries:\nxvSTNaa = Array{Array{TaylorN{Float64},1}}(undef, length(tvSTN)+1 );\nxvSTNaa[1] = x0TN\nfor ind in 2:length(tvSTN)+1\n    whatever = xvSTN[ind-1,:]\n    xvSTNaa[ind] = whatever\nend\ntvSTNaa = union([zero(tvSTN[1])], tvSTN);\n\nmyrnd  = 0:0.01:1\nnpoints = length(myrnd)\nncrosses = length(tvSTN)\nyS = Array{Float64}(undef, ncrosses+1, npoints)\npS = Array{Float64}(undef, ncrosses+1, npoints)\n\nmyrad=0.005\nξy = @. myrad * cos(2pi*myrnd)\nξp = @. myrad * sin(2pi*myrnd)\n\nfor indpoint in 1:npoints\n    yS[1,indpoint] = x0[2] + ξy[indpoint]\n    pS[1,indpoint] = x0[4] + ξp[indpoint]\n    mycond = [0.0, ξy[indpoint], 0.0, ξp[indpoint]]\n    for indS in 2:ncrosses+1\n        temp = evaluate(xvSTNaa[indS], mycond)\n        yS[indS,indpoint] = temp[2]\n        pS[indS,indpoint] = temp[4]\n    end\nend\n\npoincare_anim2 = @animate for i=1:21\n    scatter(map(x->x[i,2], xvSv), map(x->x[i,4], xvSv), marker=(:circle, stroke(0)),\n        markersize=0.01, label=\"Monte Carlo\")\n    plot!(yS[i,:], pS[i,:], width=0.1, label=\"Jet transport\")\n    xlims!(0.09,0.5)\n    ylims!(-0.11,0.11)\n    xlabel!(\"y\")\n    ylabel!(\"py\")\n    title!(\"Poincaré map: 4th-order jet transport vs Monte Carlo\")\nend\ngif(poincare_anim2, \"poincareanim2.gif\", fps = 2);\nnothing # hide(Image: Poincaré map: Jet transport vs Monte Carlo)The next animation is the same as before, adapting the scale.poincare_anim3 = @animate for i=1:21\n    scatter(map(x->x[i,2], xvSv), map(x->x[i,4], xvSv), marker=(:circle, stroke(0)),\n        markersize=0.01, label=\"Monte Carlo\")\n    plot!(yS[i,:], pS[i,:], width=0.1, label=\"Jet transport\")\n    xlabel!(\"y\")\n    ylabel!(\"py\")\n    title!(\"Poincaré map: 4th-order jet transport vs Monte Carlo\")\nend\ngif(poincare_anim3, \"poincareanim3.gif\", fps = 2);\nnothing # hide(Image: Poincaré map: Jet transport vs Monte Carlo)"
},

{
    "location": "common.html#",
    "page": "Interoperability with DifferentialEquations.jl",
    "title": "Interoperability with DifferentialEquations.jl",
    "category": "page",
    "text": ""
},

{
    "location": "common.html#diffeqinterface-1",
    "page": "Interoperability with DifferentialEquations.jl",
    "title": "Interoperability with DifferentialEquations.jl",
    "category": "section",
    "text": "Here, we show an example of interoperability between TaylorIntegration.jl and DifferentialEquations.jl. That is, how to use TaylorIntegration.jl from DifferentialEquations.jl. Below, we shall use ParameterizedFunctions.jl to define the appropriate system of ODEs. Also, we use OrdinaryDiffEq.jl, in order to compare the accuracy of TaylorIntegration.jl with respect to high-accuracy methods for non-stiff problems.The problem we will integrate in this example is the planar circular restricted three-body problem (PCR3BP, also capitalized as PCRTBP). The PCR3BP describes the motion of a body with negligible mass m_3 under the gravitational influence of two bodies with masses m_1 and m_2, such that m_1 ge m_2. It is assumed that m_3 is much smaller than the other two masses, and therefore it is simply considered as a massless test particle. The body with the greater mass m_1 is referred as the primary, and m_2 as the secondary. These bodies are together called the primaries and are assumed to describe Keplerian circular orbits about their center of mass, which is placed at the origin of the reference frame. It is further assumed that the orbit of the third body takes place in the orbital plane of the primaries. A full treatment of the PCR3BP may be found in [1].The ratio mu = m_2(m_1+m_2) is known as the mass parameter. Using mass units such that m_1+m_2=1, we have m_1=1-mu and m_2=mu. In this example, we assume the mass parameter to have a value mu=001.using TaylorIntegration, Plots\n\nμ = 0.01\nnothing # hideThe Hamiltonian for the PCR3BP in the synodic frame (i.e., a frame which rotates such that the primaries are at rest on the x axis) isbeginequation\nlabeleq-pcr3bp-hamiltonian\nH(x y p_x p_y) = frac12(p_x^2+p_y^2) - (x p_y - y p_x) + V(x y)\nendequationwherebeginequation\nlabeleq-pcr3bp-potential\nV(x y) = - frac1-musqrt(x-mu)^2+y^2 - fracmusqrt(x+1-mu)^2+y^2\nendequationis the gravitational potential associated to the primaries. The RHS of Eq. (\\ref{eq-pcr3bp-hamiltonian}) is also known as the Jacobi constant, since it is a preserved quantity of motion in the PCR3BP. We will use this property to check the accuracy of the obtained solution.V(x, y) = - (1-μ)/sqrt((x-μ)^2+y^2) - μ/sqrt((x+1-μ)^2+y^2)\nH(x, y, px, py) = (px^2+py^2)/2 - (x*py-y*px) + V(x, y)\nH(x) = H(x...)\nnothing # hideThe equations of motion for the PCR3BP arebegineqnarray\nlabeleqs-motion-pcr3bp\n    dotx = p_x + y \n    doty = p_y - x \n    dotp_x = - frac(1-mu)(x-mu)((x-mu)^2+y^2)^32 - fracmu(x+1-mu)((x+1-mu)^2+y^2)^32 + p_y \n    dotp_y = - frac(1-mu)y      ((x-mu)^2+y^2)^32 - fracmu y       ((x+1-mu)^2+y^2)^32 - p_x\nendeqnarrayWe define this system of ODEs with ParameterizedFunctions.jlusing ParameterizedFunctions\n\nf = @ode_def PCR3BP begin\n    dx = px + y\n    dy = py - x\n    dpx = - (1-μ)*(x-μ)*((x-μ)^2+y^2)^-1.5 - μ*(x+1-μ)*((x+1-μ)^2+y^2)^-1.5 + py\n    dpy = - (1-μ)*y    *((x-μ)^2+y^2)^-1.5 - μ*y      *((x+1-μ)^2+y^2)^-1.5 - px\nend μ\nnothing # hideWe shall define the initial conditions q_0 = (x_0 y_0 p_x0 p_y0) such that H(q_0) = J_0, where J_0 is a prescribed value. In order to do this, we select y_0 = p_x0 = 0 and compute the value of p_y0 for which H(q_0) = J_0 holds.We consider a value for J_0 such that the test particle is able to display close encounters with both primaries, but cannot escape to infinity. We may obtain a first approximation to the desired value of J_0 if we plot the projection of the zero-velocity curves on the x-axis.ZVC(x) =  -x^2/2 + V(x, zero(x)) # projection of the zero-velocity curves on the x-axis\n\nplot(ZVC, -2:0.001:2, label=\"zero-vel. curve\", legend=:topleft)\nplot!([-2, 2], [-1.58, -1.58], label=\"J0 = -1.58\")\nylims!(-1.7, -1.45)\nxlabel!(\"x\")\nylabel!(\"J\")\ntitle!(\"Zero-velocity curves (x-axis projection)\")Notice that the maxima in the plot correspond to the Lagrangian points L_1, L_2 and L_3; below we shall concentrate in the value J_0 = -158.J0 = -1.58\nnothing # hideWe define a function py!, which depends on the initial condition q_0 = (x_0 0 0 p_y0) and the Jacobi constant value J_0, such that it computes an adequate value p_y0 for which we have H(q_0)=J_0 and updates (in-place) the initial condition accordingly.function py!(q0, J0)\n    @assert iszero(q0[2]) && iszero(q0[3]) # q0[2] and q0[3] have to be equal to zero\n    q0[4] = q0[1] + sqrt( q0[1]^2-2( V(q0[1], q0[2])-J0 ) )\n    nothing\nend\nnothing # hideWe are now ready to generate an appropriate initial condition.q0 = [-0.8, 0.0, 0.0, 0.0]\npy!(q0, J0)\nq0We note that the value of q0 has been updated. We can check that the value of the Hamiltonian evaluated at the initial condition is indeed equal to J0.H(q0) == J0Following the tutorial of DifferentialEquations.jl, we define an ODEProblem for the integration; TaylorIntegration.jl can be used via its common interface bindings with JuliaDiffEq.tspan = (0.0, 1000.0)\np = [μ]\n\nusing TaylorIntegration # `ODEProblem` is exported by TaylorIntegration\nprob = ODEProblem(f, q0, tspan, p)We solve prob using a 25-th order Taylor method, with a local absolute tolerance epsilon_mathrmtol = 10^-20; note that TaylorIntegration.jl has to be loaded independently.solT = solve(prob, TaylorMethod(25), abstol=1e-20);Likewise, we load OrdinaryDiffEq in order to solve the same problem prob with the Vern9 method, which the DifferentialEquations.jl documentation recommends for high-accuracy (i.e., very low tolerance) integrations of non-stiff problems.using OrdinaryDiffEq\n\nsolV = solve(prob, Vern9(), abstol=1e-20); #solve `prob` with the `Vern9` methodWe plot in the x-y synodic plane the solution obtained with TaylorIntegration.jl:plot(solT, vars=(1, 2), linewidth=1)\nscatter!([μ, -1+μ], [0,0], leg=false) # positions of the primaries\nxlims!(-1+μ-0.2, 1+μ+0.2)\nylims!(-0.8, 0.8)\nxlabel!(\"x\")\nylabel!(\"y\")Note that the orbit obtained displays the expected dynamics: the test particle explores the regions surrounding both primaries, located at the red dots, without escaping to infinity. For comparison, we now plot the orbit corresponding to the solution obtained with the Vern9() integration; note that the scales are identical.plot(solV, vars=(1, 2), linewidth=1)\nscatter!([μ, -1+μ], [0,0], leg=false) # positions of the primaries\nxlims!(-1+μ-0.2, 1+μ+0.2)\nylims!(-0.8, 0.8)\nxlabel!(\"x\")\nylabel!(\"y\")We note that the orbits do not display the same qualitative features. In particular, the Vern9() integration displays an orbit which does not visit the secondary, as it was the case in the integration using Taylor\'s method, and stays far enough from m_1. The question is which integration should we trust?We can obtain a quantitative comparison of the validity of both integrations through the preservation of the Jacobi constant:ET = H.(solT.u)\nEV = H.(solV.u)\nδET = ET .- J0\nδEV = EV .- J0\nnothing # hideWe plot first the value of the Jacobi constant as function of time.plot(solT.t, H.(solT.u), label=\"TaylorIntegration.jl\")\nplot!(solV.t, H.(solV.u), label=\"Vern9()\")\nxlabel!(\"t\")\nylabel!(\"H\")Clearly, the integration with Vern9() does not conserve the Jacobi constant; actually, the fact that its value is strongly reduced leads to the artificial trapping displayed above around m_1. We notice that the loss of conservation of the Jacobi constant is actually not related to a close approach with m_1.We now plot, in log scale, the abs of the absolute error in the Jacobi constant as a function of time, for both solutions:plot(solT.t, abs.(δET), yscale=:log10, label=\"TaylorIntegration.jl\")\nplot!(solV.t, abs.(δEV), label=\"Vern9()\")\nylims!(10^-18, 10^4)\nxlabel!(\"t\")\nylabel!(\"dE\")We notice that the Jacobi constant absolute error for the TaylorIntegration.jl solution remains bounded below 5times 10^-14, despite of the fact that the solution displays many close approaches with m_2.Finally, we comment on the time spent by each integration.@time solve(prob, TaylorMethod(25), abstol=1e-20);@time solve(prob, Vern9(), abstol=1e-20);Clearly, the integration with TaylorMethod() takes much longer than that using Vern9(). Yet, as shown above, the former preserves the Jacobi constant to a high accuracy while displaying the correct dynamics; whereas the latter solution loses accuracy in the sense of not conserving the Jacobi constant, which is an important property to trust the result of the integration."
},

{
    "location": "common.html#refsPCR3BP-1",
    "page": "Interoperability with DifferentialEquations.jl",
    "title": "References",
    "category": "section",
    "text": "[1] Murray, Carl D., Stanley F. Dermott. Solar System dynamics. Cambridge University Press, 1999."
},

{
    "location": "api.html#",
    "page": "API",
    "title": "API",
    "category": "page",
    "text": ""
},

{
    "location": "api.html#Library-1",
    "page": "API",
    "title": "Library",
    "category": "section",
    "text": "CurrentModule = TaylorIntegration"
},

{
    "location": "api.html#TaylorIntegration.lyap_taylorinteg-Union{Tuple{U}, Tuple{T}, Tuple{Any,Array{U,1},T,T,Int64,T}, Tuple{Any,Array{U,1},T,T,Int64,T,Any}} where U<:Number where T<:Real",
    "page": "API",
    "title": "TaylorIntegration.lyap_taylorinteg",
    "category": "method",
    "text": "lyap_taylorinteg(f!, q0, t0, tmax, order, abstol[, f!]; maxsteps::Int=500)\n\nSimilar to taylorinteg for the calculation of the Lyapunov spectrum. Note that the number of TaylorN variables should be set previously by the user (e.g., by means of TaylorSeries.set_variables) and should be equal to the length of the vector of initial conditions q0. Otherwise, whenever length(q0) != TaylorSeries.get_numvars(), then lyap_taylorinteg throws an AssertionError. Optionally, the user may provide a Jacobian function jacobianfunc! to evaluate the current value of the Jacobian. Otherwise, the current value of the Jacobian is computed via automatic differentiation using TaylorSeries.jl.\n\n\n\n\n\n"
},

{
    "location": "api.html#TaylorIntegration.taylorinteg-Union{Tuple{U}, Tuple{T}, Tuple{Any,Any,Array{U,1},T,T,Int64,T}} where U<:Number where T<:Real",
    "page": "API",
    "title": "TaylorIntegration.taylorinteg",
    "category": "method",
    "text": "taylorinteg(f, g, x0, t0, tmax, order, abstol; kwargs... )\n\nRoot-finding method of taylorinteg. Given a function g(t, x, dx), called the event function, taylorinteg checks for the occurrence of a root of g evaluated at the solution; that is, it checks for the occurrence of an event or condition specified by g=0. Then, taylorinteg attempts to find that root (or event, or crossing) by performing a Newton-Raphson process. When called with the eventorder=n keyword argument, taylorinteg searches for the roots of the n-th derivative of g, which is computed via automatic differentiation.\n\nmaxsteps is the maximum number of allowed time steps; eventorder is the order of the derivatives of g whose roots the user is interested in finding; newtoniter is the maximum number of Newton-Raphson iterations per detected root; nrabstol is the allowed tolerance for the Newton-Raphson process.\n\nThe current keyword arguments are maxsteps=500, eventorder=0, newtoniter=10, and nrabstol=eps(T), where T is the common type of t0, tmax and abstol.\n\nFor more details about conventions in taylorinteg, please see taylorinteg.\n\nExamples:\n\n    using TaylorIntegration\n\n    function pendulum!(t, x, dx)\n        dx[1] = x[2]\n        dx[2] = -sin(x[1])\n        nothing\n    end\n\n    g(t, x, dx) = x[2]\n\n    x0 = [1.3, 0.0]\n\n    # find the roots of `g` along the solution\n    tv, xv, tvS, xvS, gvS = taylorinteg(pendulum!, g, x0, 0.0, 22.0, 28, 1.0E-20);\n\n    # find the roots of the 2nd derivative of `g` along the solution\n    tv, xv, tvS, xvS, gvS = taylorinteg(pendulum!, g, x0, 0.0, 22.0, 28, 1.0E-20; eventorder=2);\n\n\n\n\n\n"
},

{
    "location": "api.html#TaylorIntegration.taylorinteg-Union{Tuple{U}, Tuple{T}, Tuple{Any,U,Union{AbstractRange{T}, Array{T,1}},Int64,T}} where U<:Number where T<:Real",
    "page": "API",
    "title": "TaylorIntegration.taylorinteg",
    "category": "method",
    "text": "taylorinteg(f, x0, t0, trange, order, abstol; keyword... )\n\nGeneral-purpose Taylor integrator for the explicit ODE dotx=f(tx) with initial condition specified by x0::{T<:Number} or x0::Vector{T} at time t0. It returns a vector (of type typeof(x0)) with the computed values of the dependent variable(s), evaluated only at the times specified by the range trange. The integration stops at tmax=trange[end] (in which case the last returned values are t_max, x(t_max)), or else when the number of computed time steps is larger than maxsteps.\n\nThe integration uses polynomial expansions on the independent variable of order order; the parameter abstol serves to define the time step using the last two Taylor coefficients of the expansions. Make sure you use a large enough order to assure convergence.\n\nThe current keyword argument is maxsteps=500.\n\nExamples:\n\nOne dependent variable: The function f defines the equation of motion.\n\n    using TaylorIntegration\n\n    f(t, x) = x^2\n\n    xv = taylorinteg(f, 3.0, 0.0:0.001:0.3, 25, 1.0e-20, maxsteps=100 )\n\nMany (two or more) dependent variable: The function f! defines the   equation of motion.\n\n    using TaylorIntegration\n\n    function f!(t, x, dx)\n        for i in eachindex(x)\n            dx[i] = x[i]^2\n        end\n    end\n\n    xv = taylorinteg(f!, [3.0, 3.0], 0.0:0.001:0.3, 25, 1.0e-20, maxsteps=100 )\n\nNote that f! updates (mutates) the pre-allocated vector dx.\n\nJet transport for the simple pendulum.\n\n    using TaylorIntegration # TaylorSeries is reexported automatically\n\n    function pendulum!(t, x, dx) #the simple pendulum ODE\n        dx[1] = x[2]\n        dx[2] = -sin(x[1])\n    end\n\n    p = set_variables(\"ξ\", numvars=2, order=5) #TaylorN set-up, order 5\n    q0 = [1.3, 0.0]    # initial conditions\n    q0TN = q0 + p      # parametrization of a neighbourhood around q0\n    tr = 0.0:0.125:6pi\n\n    @time xv = taylorinteg(pendulum!, q0TN, tr, 28, 1e-20, maxsteps=100);\n\nNote that the initial conditions q0TN are of type TaylorN{Float64}.\n\n\n\n\n\n"
},

{
    "location": "api.html#TaylorIntegration.taylorinteg-Union{Tuple{V}, Tuple{U}, Tuple{T}, Tuple{S}, Tuple{Any,S,T,U,Int64,V}} where V<:Real where U<:Real where T<:Real where S<:Number",
    "page": "API",
    "title": "TaylorIntegration.taylorinteg",
    "category": "method",
    "text": "taylorinteg(f, x0, t0, tmax, order, abstol; keyword... )\n\nGeneral-purpose Taylor integrator for the explicit ODE dotx=f(tx) with initial condition specified by x0 at time t0. The initial condition x0 may be of type T<:Number or a Vector{T}, with T including TaylorN{T}; the latter case is of interest for jet transport applications.\n\nIt returns a vector with the values of time (independent variable), and a vector (of type typeof(x0)) with the computed values of the dependent variable(s). The integration stops when time is larger than tmax (in which case the last returned values are t_max, x(t_max)), or else when the number of saved steps is larger than maxsteps.\n\nThe integration uses polynomial expansions on the independent variable of order order; the parameter abstol serves to define the time step using the last two Taylor coefficients of the expansions. Make sure you use a large enough order to assure convergence.\n\nThe current keyword argument is maxsteps=500.\n\nExamples:\n\nOne dependent variable: The function f defines the equation of motion.\n\n    using TaylorIntegration\n\n    f(t, x) = x^2\n\n    tv, xv = taylorinteg(f, 3, 0.0, 0.3, 25, 1.0e-20, maxsteps=100 )\n\nMany (two or more) dependent variable: The function f! defines   the equation of motion.\n\n    using TaylorIntegration\n\n    function f!(t, x, dx)\n        for i in eachindex(x)\n            dx[i] = x[i]^2\n        end\n    end\n\n    tv, xv = taylorinteg(f!, [3.0,3.0], 0.0, 0.3, 25, 1.0e-20, maxsteps=100 )\n\nNote that f! updates (mutates) the pre-allocated vector dx.\n\n\n\n\n\n"
},

{
    "location": "api.html#Exported-functions-1",
    "page": "API",
    "title": "Exported functions",
    "category": "section",
    "text": "Modules = [TaylorIntegration]\nPrivate = false"
},

{
    "location": "api.html#TaylorIntegration.findroot!-NTuple{19,Any}",
    "page": "API",
    "title": "TaylorIntegration.findroot!",
    "category": "method",
    "text": "findroot!(g, t, x, dx, g_val_old, g_val, eventorder, tvS, xvS, gvS,\n    t0, δt_old, x_dx, x_dx_val, g_dg, g_dg_val, nrabstol,\n    newtoniter, nevents) -> nevents\n\nInternal root-finding subroutine, based on Newton-Raphson process. If there is a crossing, then the crossing data is stored in tvS, xvS and gvS and nevents, the number of events/crossings, is updated. g is the event function, t is a Taylor1 polynomial which represents the independent variable; x is an array of Taylor1 variables which represent the vector of dependent variables; dx is an array of Taylor1 variables which represent the LHS of the ODE; g_val_old is the last-before-current value of event function g; g_val is the current value of the event function g; eventorder is the order of the derivative of g whose roots the user is interested in finding; tvS stores the surface-crossing instants; xvS stores the value of the solution at each of the crossings; gvS stores the values of the event function g (or its eventorder-th derivative) at each of the crossings; t0 is the current time; δt_old is the last time-step size; x_dx, x_dx_val, g_dg, g_dg_val are auxiliary variables; nrabstol is the Newton-Raphson process tolerance; newtoniter is the maximum allowed number of Newton-Raphson iteration; nevents is the current number of detected events/crossings.\n\n\n\n\n\n"
},

{
    "location": "api.html#TaylorIntegration.jetcoeffs!-Union{Tuple{U}, Tuple{T}, Tuple{Any,Taylor1{T},AbstractArray{Taylor1{U},1},AbstractArray{Taylor1{U},1},AbstractArray{Taylor1{U},1}}} where U<:Number where T<:Real",
    "page": "API",
    "title": "TaylorIntegration.jetcoeffs!",
    "category": "method",
    "text": "jetcoeffs!(eqsdiff!, t, x, dx, xaux)\n\nMutates x in-place using the recursion relation of the derivatives obtained from the differential equations dotx=dxdt=f(tx).\n\neqsdiff! is the function defining the RHS of the ODE, x contains the Taylor1 expansion of the dependent variables and t is the independent variable. See taylorinteg for examples and structure of eqsdiff!. Note that x is of type Vector{Taylor1{U}} where U<:Number; t is of type Taylor1{T} where T<:Real. In this case, two auxiliary containers dx and xaux (both of the same type as x) are needed to avoid allocations.\n\nInitially, x contains only the 0-th order Taylor coefficient of the current system state (the initial conditions), and jetcoeffs! computes recursively the high-order derivates back into x.\n\n\n\n\n\n"
},

{
    "location": "api.html#TaylorIntegration.jetcoeffs!-Union{Tuple{U}, Tuple{T}, Tuple{Any,Taylor1{T},Taylor1{U}}} where U<:Number where T<:Real",
    "page": "API",
    "title": "TaylorIntegration.jetcoeffs!",
    "category": "method",
    "text": "jetcoeffs!(eqsdiff, t, x)\n\nReturns an updated x using the recursion relation of the derivatives obtained from the differential equations dotx=dxdt=f(tx).\n\neqsdiff is the function defining the RHS of the ODE, x contains the Taylor1 expansion of the dependent variable(s) and t is the independent variable. See taylorinteg for examples and structure of eqsdiff. Note that x is of type Taylor1{U} where U<:Number; t is of type Taylor1{T} where T<:Real.\n\nInitially, x contains only the 0-th order Taylor coefficient of the current system state (the initial conditions), and jetcoeffs! computes recursively the high-order derivates back into x.\n\n\n\n\n\n"
},

{
    "location": "api.html#TaylorIntegration.lyap_jetcoeffs!-Union{Tuple{S}, Tuple{T}, Tuple{Taylor1{T},AbstractArray{Taylor1{S},1},AbstractArray{Taylor1{S},1},Array{Taylor1{S},2},Array{Taylor1{S},3}}} where S<:Number where T<:Real",
    "page": "API",
    "title": "TaylorIntegration.lyap_jetcoeffs!",
    "category": "method",
    "text": "lyap_jetcoeffs!(t, x, dx, jac, varsaux)\n\nSimilar to jetcoeffs! for the calculation of the Lyapunov spectrum. Updates only the elements of x which correspond to the solution of the 1st-order variational equations dotxi=J cdot xi, where J is the Jacobian matrix, i.e., the linearization of the equations of motion. jac is the Taylor expansion of J wrt the independent variable, around the current initial condition. varsaux is an auxiliary array of type Array{eltype(jac),3} to avoid allocations. Calling this method assumes that jac has been computed previously using stabilitymatrix!.\n\n\n\n\n\n"
},

{
    "location": "api.html#TaylorIntegration.lyap_taylorstep!-Union{Tuple{U}, Tuple{T}, Tuple{Any,Taylor1{T},Array{Taylor1{U},1},Array{Taylor1{U},1},Array{Taylor1{U},1},Array{TaylorN{Taylor1{U}},1},Array{TaylorN{Taylor1{U}},1},Array{Taylor1{U},2},T,T,Array{U,1},Int64,T,Array{TaylorN{Taylor1{U}},1},Array{Taylor1{U},3}}, Tuple{Any,Taylor1{T},Array{Taylor1{U},1},Array{Taylor1{U},1},Array{Taylor1{U},1},Array{TaylorN{Taylor1{U}},1},Array{TaylorN{Taylor1{U}},1},Array{Taylor1{U},2},T,T,Array{U,1},Int64,T,Array{TaylorN{Taylor1{U}},1},Array{Taylor1{U},3},Any}} where U<:Number where T<:Real",
    "page": "API",
    "title": "TaylorIntegration.lyap_taylorstep!",
    "category": "method",
    "text": "lyap_taylorstep!(f!, t, x, dx, xaux, δx, dδx, jac, t0, t1, x0, order, abstol, _δv, varsaux[, jacobianfunc!])\n\nSimilar to taylorstep! for the calculation of the Lyapunov spectrum. jac is the Taylor expansion (wrt the independent variable) of the linearization of the equations of motion, i.e, the Jacobian. xaux, δx, dδx, varsaux and _δv are auxiliary vectors. Optionally, the user may provide a Jacobian function jacobianfunc! to compute jac. Otherwise, jac is computed via automatic differentiation using TaylorSeries.jl.\n\n\n\n\n\n"
},

{
    "location": "api.html#TaylorIntegration.nrconvergencecriterion-Union{Tuple{T}, Tuple{U}, Tuple{U,T,Int64,Int64}} where T<:Real where U<:Number",
    "page": "API",
    "title": "TaylorIntegration.nrconvergencecriterion",
    "category": "method",
    "text": "nrconvergencecriterion(g_val, nrabstol::T, nriter::Int, newtoniter::Int) where {T<:Real}\n\nA rudimentary convergence criterion for the Newton-Raphson root-finding process. g_val may be either a Real, Taylor1{T} or a TaylorN{T}, where T<:Real. Returns true if: 1) the absolute value of g_val, the event function g evaluated at the current estimated root by the Newton-Raphson process, is less than the nrabstol tolerance; and 2) the number of iterations nriter of the Newton-Raphson process is less than the maximum allowed number of iterations, newtoniter; otherwise, returns false.\n\n\n\n\n\n"
},

{
    "location": "api.html#TaylorIntegration.stabilitymatrix!-Union{Tuple{U}, Tuple{T}, Tuple{Any,Taylor1{T},Array{Taylor1{U},1},Array{TaylorN{Taylor1{U}},1},Array{TaylorN{Taylor1{U}},1},Array{Taylor1{U},2},Array{TaylorN{Taylor1{U}},1}}, Tuple{Any,Taylor1{T},Array{Taylor1{U},1},Array{TaylorN{Taylor1{U}},1},Array{TaylorN{Taylor1{U}},1},Array{Taylor1{U},2},Array{TaylorN{Taylor1{U}},1},Any}} where U<:Number where T<:Real",
    "page": "API",
    "title": "TaylorIntegration.stabilitymatrix!",
    "category": "method",
    "text": "stabilitymatrix!(eqsdiff!, t, x, δx, dδx, jac, _δv[, jacobianfunc!])\n\nUpdates the matrix jac::Matrix{Taylor1{U}} (linearized equations of motion) computed from the equations of motion (eqsdiff!), at time t at x; x is of type Vector{Taylor1{U}}, where U<:Number. δx, dδx and _δv are auxiliary arrays of type Vector{TaylorN{Taylor1{U}}} to avoid allocations. Optionally, the user may provide a Jacobian function jacobianfunc! to compute jac. Otherwise, jac is computed via automatic differentiation using TaylorSeries.jl.\n\n\n\n\n\n"
},

{
    "location": "api.html#TaylorIntegration.stepsize-Union{Tuple{U}, Tuple{T}, Tuple{Taylor1{U},T}} where U<:Number where T<:Real",
    "page": "API",
    "title": "TaylorIntegration.stepsize",
    "category": "method",
    "text": "stepsize(x, epsilon) -> h\n\nReturns a maximum time-step for a the Taylor expansion x using a prescribed absolute tolerance epsilon and the last two Taylor coefficients of (each component of) x.\n\nNote that x is of type Taylor1{T} or Vector{Taylor1{T}}, including also the cases Taylor1{TaylorN{T}} and Vector{Taylor1{TaylorN{T}}}.\n\n\n\n\n\n"
},

{
    "location": "api.html#TaylorIntegration.surfacecrossing-Union{Tuple{T}, Tuple{Taylor1{T},Taylor1{T},Int64}} where T<:Number",
    "page": "API",
    "title": "TaylorIntegration.surfacecrossing",
    "category": "method",
    "text": "surfacecrossing(g_old, g_now, eventorder::Int)\n\nDetect if the solution crossed a root of event function g. g_old represents the last-before-current value of event function g; g_now represents the current value of event function g; eventorder is the order of the derivative of the event function g whose root we are trying to find. Returns true if g_old and g_now have different signs (i.e., if one is positive and the other one is negative); otherwise returns false.\n\n\n\n\n\n"
},

{
    "location": "api.html#TaylorIntegration.taylorstep!-Union{Tuple{U}, Tuple{T}, Tuple{Any,Taylor1{T},Array{Taylor1{U},1},Array{Taylor1{U},1},Array{Taylor1{U},1},T,T,Array{U,1},Int64,T}} where U<:Number where T<:Real",
    "page": "API",
    "title": "TaylorIntegration.taylorstep!",
    "category": "method",
    "text": "taylorstep!(f!, t, x, dx, xaux, t0, t1, x0, order, abstol) -> δt\n\nOne-step Taylor integration for the ODE dotx=dxdt=f(t x) with initial conditions x(t_0)=x_0, computed from t0 up to t1, returning the time-step of the actual integration carried out and updating (in-place) x0.\n\nHere, f! is the function defining the RHS of the ODE (see taylorinteg for examples and structure of f!), t is the independent variable, x contains the Taylor expansion of the dependent variables, x0 corresponds to the initial (and updated) dependent variables and is of type Vector{Taylor1{T<:Number}}, order is the degree used for the Taylor1 polynomials during the integration and abstol is the absolute tolerance used to determine the time step of the integration.  dx and xaux, both of the same type as x0, are needed to avoid allocations.\n\n\n\n\n\n"
},

{
    "location": "api.html#TaylorIntegration.taylorstep!-Union{Tuple{U}, Tuple{T}, Tuple{Any,Taylor1{T},Taylor1{U},T,T,U,Int64,T}} where U<:Number where T<:Real",
    "page": "API",
    "title": "TaylorIntegration.taylorstep!",
    "category": "method",
    "text": "taylorstep!(f, t, x, t0, t1, x0, order, abstol) -> δt, x0\n\nOne-step Taylor integration for the ODE dotx=dxdt=f(t x) with initial conditions x(t_0)=x_0, computed from t0 up to t1. Returns the time-step of the actual integration carried out and the updated value of x0.\n\nHere, f is the function defining the RHS of the ODE (see taylorinteg for examples and structure of f), t is the independent variable, x contains the Taylor expansion of the dependent variable,x0 is the initial value of the dependent variable, order is the degree  used for the Taylor1 polynomials during the integration and abstol is the absolute tolerance used to determine the time step of the integration. Note that x0 is of type Taylor1{T<:Number} or Taylor1{TaylorN{T}}. If the time step is larger than t1-t0, that difference is used as the time step.\n\n\n\n\n\n"
},

{
    "location": "api.html#Internal-1",
    "page": "API",
    "title": "Internal",
    "category": "section",
    "text": "Modules = [TaylorIntegration]\nPublic = false"
},

]}
