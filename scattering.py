import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt


def solve(mesh, k, a0, a1, b0, b1, g, neumann=False,
          quad_deg=None, beta=None, annular=False):
    # Function space
    W = fd.FunctionSpace(mesh, "CG", 1)
    p = fd.TrialFunction(W)
    q = fd.conj(fd.TestFunction(W))
    solution = fd.Function(W, name="sol")

    # Coefficient functions
    X = fd.SpatialCoordinate(mesh)
    if beta is None:
        beta = 1
    sigma_x = beta / k / (a1 - abs(X[0]))
    sigma_y = beta / k / (b1 - abs(X[1]))
    gamma_x = 1 + 1j * sigma_x
    gamma_y = 1 + 1j * sigma_y

    # Boundary data
    if neumann:
        bcs = [fd.DirichletBC(W, 0., 3)]
    else:
        bcs = [fd.DirichletBC(W, g, 1),
               fd.DirichletBC(W, 0., 3)]

    # Bilinear form
    if annular:
        Omega_F = fd.dx(1) + fd.dx(5)
    else:
        Omega_F = fd.dx(1)
    a = (fd.dot(fd.grad(p), fd.grad(q)) - k**2 * p * q) * Omega_F \
        + (1 / gamma_x * p.dx(0) * q.dx(0)
            + gamma_x * p.dx(1) * q.dx(1)
            - k**2 * gamma_x * p * q) * fd.dx(2) \
        + (gamma_y * p.dx(0) * q.dx(0)
            + 1 / gamma_y * p.dx(1) * q.dx(1)
            - k**2 * gamma_y * p * q) * fd.dx(3) \
        + (gamma_y / gamma_x * p.dx(0) * q.dx(0)
            + gamma_x / gamma_y * p.dx(1) * q.dx(1)
            - k**2 * gamma_x * gamma_y * p * q) * fd.dx(4)

    # Linear form
    L = g * q * fd.ds(1) if neumann else 0

    # Solution
    fcp = {}
    if quad_deg is not None:
        fcp["quadrature_degree"] = quad_deg
    problem = fd.LinearVariationalProblem(
        a, L, solution, bcs, form_compiler_parameters=fcp)
    solver = fd.LinearVariationalSolver(problem)
    solver.solve()
    return solution


def compute_error(u, uh, relative=True, norm="l2",
                  quad_rule=None, quad_deg=None):
    fcp = {}
    if quad_rule is not None:
        fcp["quadrature_rule"] = quad_rule
    if quad_deg is not None:
        fcp["quadrature_degree"] = quad_deg
    if norm == "l2":
        def form(h): return fd.inner(h, h)
    elif norm == "h1_semi":
        def form(h): return fd.inner(fd.grad(h), fd.grad(h))
    elif norm == "h1":
        def form(h): return fd.inner(h, h) + fd.inner(fd.grad(h), fd.grad(h))

    err = fd.assemble(form(u - uh) * fd.dx(1),
                      form_compiler_parameters=fcp)**0.5
    if relative:
        err /= fd.assemble(form(u) * fd.dx(1),
                           form_compiler_parameters=fcp)**0.5
    return err.real


def far_field(k, u_s, theta, inc=0, boundary=1):
    mesh = u_s.function_space().mesh()
    x = fd.Constant((np.cos(theta), np.sin(theta)))
    y = fd.SpatialCoordinate(mesh)
    n = fd.FacetNormal(mesh)
    u = u_s + inc
    if boundary == 1:
        res = np.exp(1j*np.pi/4) / np.sqrt(8*np.pi*k) * fd.assemble(
            fd.dot(-1j * k * u * x - fd.grad(u), -n)
            * fd.exp(-1j * k * fd.dot(x, y)) * fd.ds(1))
    else:
        res = np.exp(1j*np.pi/4) / np.sqrt(8*np.pi*k) * fd.assemble(
            fd.dot(-1j * k * u * x - fd.avg(fd.grad(u)), n('+'))
            * fd.exp(-1j * k * fd.dot(x, y)) * fd.dS(boundary))
    return res


def far_field_vol(k, u_s, theta, R0, R1, inc=0):
    mesh = u_s.function_space().mesh()
    x = fd.Constant((np.cos(theta), np.sin(theta)))
    y = fd.SpatialCoordinate(mesh)
    r = fd.real(fd.sqrt(fd.inner(y, y)))
    psi = (1 - fd.cos((r - R0) / (R1 - R0) * fd.pi)) / 2
    u = u_s + inc

    fcp = {"quadrature_degree": 4}
    res = np.exp(1j*np.pi/4) / np.sqrt(8*np.pi*k) * fd.assemble(
        u * (fd.div(fd.grad(psi)) - 2j * k * fd.dot(x, fd.grad(psi)))
        * fd.exp(-1j * k * fd.dot(x, y)) * fd.dx(5),
        form_compiler_parameters=fcp)
    return res


def plot_mesh(m):
    fig, axes = plt.subplots()
    interior_kw = {"linewidths": 0.2}
    fd.triplot(m, axes=axes, interior_kw=interior_kw)
    axes.set_aspect("equal")
    axes.legend()


def plot_field(u, a0, a1, b0, b1):
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)

    plot1 = fd.tripcolor(u, complex_component="real", axes=ax1)
    ax1.set_aspect("equal")
    ax1.set_title("Real part")
    ax1.set_xlim(-a1, a1)
    ax1.set_ylim(-b1, b1)
    ax1.add_patch(
        plt.Rectangle((-a0, -b0), 2*a0, 2*b0, color='w', fill=False))
    fig.colorbar(plot1, shrink=0.5, ax=ax1)

    plot2 = fd.tripcolor(u, complex_component="imag", axes=ax2)
    ax2.set_aspect("equal")
    ax2.set_title("Imaginary part")
    ax2.set_xlim(-a1, a1)
    ax2.set_ylim(-b1, b1)
    ax2.add_patch(
        plt.Rectangle((-a0, -b0), 2*a0, 2*b0, color='w', fill=False))
    fig.colorbar(plot2, shrink=0.5, ax=ax2)


def plot_far_field(k, u, inc=0, boundary=1):
    theta = np.linspace(0, 2 * np.pi, 100)
    u_inf = []
    for t in list(theta):
        u_inf.append(far_field(k, u, t, inc=inc, boundary=boundary))
    u_inf = np.array(u_inf)

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'},
                                   constrained_layout=True)
    ax1.plot(theta, u_inf.real)
    ax1.set_title("Real part")
    ax1.set_rlabel_position(90)
    ax1.grid(True)
    ax2.plot(theta, u_inf.imag)
    ax2.set_title("Imaginary part")
    ax2.set_rlabel_position(90)
    ax2.grid(True)
    fig.suptitle("Boundary-based formula")


def plot_far_field_vol(k, u, R0, R1, inc=0):
    theta = np.linspace(0, 2 * np.pi, 100)
    u_inf = []
    for t in list(theta):
        u_inf.append(far_field_vol(k, u, t, R0, R1, inc=inc))
    u_inf = np.array(u_inf)

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'},
                                   constrained_layout=True)
    ax1.plot(theta, u_inf.real)
    ax1.set_title("Real part")
    ax1.set_rlabel_position(90)
    ax1.grid(True)
    ax2.plot(theta, u_inf.imag)
    ax2.set_title("Imaginary part")
    ax2.set_rlabel_position(90)
    ax2.grid(True)
    fig.suptitle("Volume-based formula")
