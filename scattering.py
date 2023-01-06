import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt


def solve(mesh, k, a0, a1, b0, b1, g, neumann=False,
          quad_deg=None, beta=None, annular=False):
    # Function space
    W = fd.VectorFunctionSpace(mesh, "CG", 1)
    p = fd.TrialFunction(W)
    q = fd.TestFunction(W)
    solution = fd.Function(W, name="sol")

    # Coefficient functions
    X = fd.SpatialCoordinate(mesh)
    if beta is None:
        beta = 1
    sigma_x = beta / k / (a1 - abs(X[0]))
    sigma_y = beta / k / (b1 - abs(X[1]))
    c1x = (1 / (1 + sigma_x**2), -sigma_x / (1 + sigma_x**2))
    c2x = (1, sigma_x)
    c3x = (1, sigma_x)
    c1y = (1, sigma_y)
    c2y = (1 / (1 + sigma_y**2), -sigma_y / (1 + sigma_y**2))
    c3y = (1, sigma_y)
    c1xy = ((1 + sigma_x * sigma_y) / (1 + sigma_x**2),
            (sigma_y - sigma_x) / (1 + sigma_x**2))
    c2xy = ((1 + sigma_x * sigma_y) / (1 + sigma_y**2),
            (sigma_x - sigma_y) / (1 + sigma_y**2))
    c3xy = (1 - sigma_x * sigma_y, sigma_x + sigma_y)

    # Boundary data
    if neumann:
        bcs = [fd.DirichletBC(W, (0., 0), 3)]
    else:
        bcs = [fd.DirichletBC(W, g, 1),
               fd.DirichletBC(W, (0., 0.), 3)]

    # Bilinear form
    if annular:
        Omega_F = fd.dx(1) + fd.dx(5)
    else:
        Omega_F = fd.dx(1)
    px, py = p.dx(0), p.dx(1)
    qx, qy = q.dx(0), q.dx(1)
    a = (inner(px, qx) + inner(py, qy) - k**2 * inner(p, q)) * Omega_F\
        + (inner(px, qx, c=c1x) + inner(py, qy, c=c2x)
            - k**2 * inner(p, q, c=c3x)) * fd.dx(2)\
        + (inner(px, qx, c=c1y) + inner(py, qy, c=c2y)
            - k**2 * inner(p, q, c=c3y)) * fd.dx(3)\
        + (inner(px, qx, c=c1xy) + inner(py, qy, c=c2xy)
            - k**2 * inner(p, q, c=c3xy)) * fd.dx(4)

    # Linear form
    L = (inner(g, q)) * fd.ds(1) if neumann else 0

    # Solution
    fcp = {}
    if quad_deg is not None:
        fcp["quadrature_degree"] = quad_deg
    problem = fd.LinearVariationalProblem(
        a, L, solution, bcs, form_compiler_parameters=fcp)
    solver = fd.LinearVariationalSolver(problem)
    solver.solve()
    return solution


def inner(x, y, c=None):
    x_re, x_im = x
    y_re, y_im = y
    if c:
        c_re, c_im = c
        res_re = c_re * x_re * y_re + c_re * x_im * y_im + c_im * x_re * y_im\
            - c_im * x_im * y_re
        res_im = -c_re * x_re * y_im + c_re * x_im * y_re + c_im * x_re * y_re\
            + c_im * x_im * y_im
    else:
        res_re = x_re * y_re + x_im * y_im
        res_im = -x_re * y_im + x_im * y_re
    return res_re + res_im


def dot(x, y):
    x_re, x_im = x
    y_re, y_im = y
    res_re = x_re * y_re - x_im * y_im
    res_im = x_re * y_im + x_im * y_re
    return res_re, res_im


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
        def form(h):
            return fd.inner(h.dx(0), h.dx(0)) + fd.inner(h.dx(1), h.dx(1))
    elif norm == "h1":
        def form(h):
            return fd.inner(h, h)\
                + fd.inner(h.dx(0), h.dx(0)) + fd.inner(h.dx(1), h.dx(1))

    err = fd.assemble(form(u - uh) * fd.dx(1),
                      form_compiler_parameters=fcp)**0.5
    if relative:
        err /= fd.assemble(form(u) * fd.dx(1),
                           form_compiler_parameters=fcp)**0.5
    return err


def far_field(k, u_s, theta, inc=None, boundary=1):
    mesh = u_s.function_space().mesh()
    x = fd.Constant((np.cos(theta), np.sin(theta)))
    y = fd.SpatialCoordinate(mesh)
    n = fd.FacetNormal(mesh)
    u = u_s + inc if inc else u_s

    phi = fd.pi / 4 - k * fd.inner(x, y)
    f = (fd.cos(phi), fd.sin(phi))
    if boundary == 1:
        g = (fd.inner(k * u[1] * x - fd.grad(u[0]), -n),
             fd.inner(-k * u[0] * x - fd.grad(u[1]), -n))
        h = dot(f, g)
        res_re = 1 / np.sqrt(8*np.pi*k) * fd.assemble(h[0] * fd.ds(1))
        res_im = 1 / np.sqrt(8*np.pi*k) * fd.assemble(h[1] * fd.ds(1))
    else:
        g = (fd.inner(k * u[1] * x - fd.avg(fd.grad(u[0])), n('+')),
             fd.inner(-k * u[0] * x - fd.avg(fd.grad(u[1])), n('+')))
        h = dot(f, g)
        res_re = 1 / np.sqrt(8*np.pi*k) * fd.assemble(h[0] * fd.dS(boundary))
        res_im = 1 / np.sqrt(8*np.pi*k) * fd.assemble(h[1] * fd.dS(boundary))
    return res_re, res_im


def far_field_vol(k, u_s, theta, R0, R1, inc=None):
    mesh = u_s.function_space().mesh()
    x = fd.Constant((np.cos(theta), np.sin(theta)))
    y = fd.SpatialCoordinate(mesh)
    r = fd.sqrt(fd.dot(y, y))
    psi = (1 - fd.cos((r - R0) / (R1 - R0) * fd.pi)) / 2
    u = u_s + inc if inc else u_s

    fcp = {"quadrature_degree": 4}
    phi = fd.pi / 4 - k * fd.inner(x, y)
    f = (fd.cos(phi), fd.sin(phi))
    g = dot(u, (fd.div(fd.grad(psi)), -2 * k * fd.dot(x, fd.grad(psi))))
    h = dot(f, g)
    res_re = 1 / np.sqrt(8*np.pi*k) *\
        fd.assemble(h[0] * fd.dx(5), form_compiler_parameters=fcp)
    res_im = 1 / np.sqrt(8*np.pi*k) *\
        fd.assemble(h[1] * fd.dx(5), form_compiler_parameters=fcp)
    return res_re, res_im


def plot_mesh(m):
    fig, axes = plt.subplots()
    interior_kw = {"linewidths": 0.2}
    fd.triplot(m, axes=axes, interior_kw=interior_kw)
    axes.set_aspect("equal")
    axes.legend()


def plot_field(u, a0, a1, b0, b1):
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)

    plot1 = fd.tripcolor(u.sub(0), axes=ax1)
    ax1.set_aspect("equal")
    ax1.set_title("Real part")
    ax1.set_xlim(-a1, a1)
    ax1.set_ylim(-b1, b1)
    ax1.add_patch(
        plt.Rectangle((-a0, -b0), 2*a0, 2*b0, color='w', fill=False))
    fig.colorbar(plot1, shrink=0.5, ax=ax1)

    plot2 = fd.tripcolor(u.sub(1), axes=ax2)
    ax2.set_aspect("equal")
    ax2.set_title("Imaginary part")
    ax2.set_xlim(-a1, a1)
    ax2.set_ylim(-b1, b1)
    ax2.add_patch(
        plt.Rectangle((-a0, -b0), 2*a0, 2*b0, color='w', fill=False))
    fig.colorbar(plot2, shrink=0.5, ax=ax2)


def plot_far_field(k, u, inc=None, boundary=1):
    theta = np.linspace(0, 2 * np.pi, 100)
    u_inf = []
    for t in list(theta):
        u_inf.append(far_field(k, u, t, inc=inc, boundary=boundary))
    u_inf = np.array(u_inf)

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'},
                                   constrained_layout=True)
    ax1.plot(theta, u_inf[:, 0])
    ax1.set_title("Real part")
    ax1.set_rlabel_position(90)
    ax1.grid(True)
    ax2.plot(theta, u_inf[:, 1])
    ax2.set_title("Imaginary part")
    ax2.set_rlabel_position(90)
    ax2.grid(True)
    fig.suptitle("Boundary-based formula")


def plot_far_field_vol(k, u, R0, R1, inc=None):
    theta = np.linspace(0, 2 * np.pi, 100)
    u_inf = []
    for t in list(theta):
        u_inf.append(far_field_vol(k, u, t, R0, R1, inc=inc))
    u_inf = np.array(u_inf)

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'},
                                   constrained_layout=True)
    ax1.plot(theta, u_inf[:, 0])
    ax1.set_title("Real part")
    ax1.set_rlabel_position(90)
    ax1.grid(True)
    ax2.plot(theta, u_inf[:, 1])
    ax2.set_title("Imaginary part")
    ax2.set_rlabel_position(90)
    ax2.grid(True)
    fig.suptitle("Volume-based formula")
