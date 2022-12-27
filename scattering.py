import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt


def solve(mesh, k, a1, b1, g_fct, neumann=False):
    # Function space
    W = fd.VectorFunctionSpace(mesh, "CG", 1)
    p = fd.TrialFunction(W)
    q = fd.TestFunction(W)
    s = fd.Function(W)
    g = fd.Function(W)

    # Coefficient functions
    X = fd.SpatialCoordinate(mesh)
    sigma_x = 1 / k / (a1 - abs(X[0]))
    sigma_y = 1 / k / (b1 - abs(X[1]))
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
    g = interpolate(W, g_fct)
    if neumann:
        bcs = [fd.DirichletBC(W, (0., 0), 3)]
    else:
        bcs = [fd.DirichletBC(W, g, 1),
               fd.DirichletBC(W, (0., 0.), 3)]

    # Bilinear form
    px, py = p.dx(0), p.dx(1)
    qx, qy = q.dx(0), q.dx(1)
    a = (prod2(px, qx) + prod2(py, qy) - k**2 * prod2(p, q)) * fd.dx(1) \
        + (prod3(c1x, px, qx) + prod3(c2x, py, qy)
            - k**2 * prod3(c3x, p, q)) * fd.dx(2) \
        + (prod3(c1y, px, qx) + prod3(c2y, py, qy)
            - k**2 * prod3(c3y, p, q)) * fd.dx(3) \
        + (prod3(c1xy, px, qx) + prod3(c2xy, py, qy)
            - k**2 * prod3(c3xy, p, q)) * fd.dx(4)

    # Linear form
    if neumann:
        L = (prod2(g, q)) * fd.ds(1)
    else:
        L = fd.Constant(0) * q[0] * fd.dx

    # Solution
    fd.solve(a == L, s, bcs=bcs)
    return s


def interpolate(W, v_fct):
    v = fd.Function(W)
    mesh = W.mesh()
    X = fd.interpolate(mesh.coordinates, W).dat.data_ro
    v_data = v_fct(X)
    v.sub(0).dat.data[:] = v_data.real
    v.sub(1).dat.data[:] = v_data.imag
    return v


def prod2(x, y, split=False):
    x_re, x_im = x
    y_re, y_im = y
    res_re = x_re * y_re - x_im * y_im
    res_im = x_re * y_im + x_im * y_re
    if split:
        return res_re, res_im
    else:
        return res_re + res_im


def prod3(x, y, z, split=False):
    x_re, x_im = x[0], x[1]
    y_re, y_im = y[0], y[1]
    z_re, z_im = z[0], z[1]
    res_re = x_re * y_re * z_re - x_re * y_im * z_im - x_im * y_re * z_im \
        - x_im * y_im * z_re
    res_im = x_re * y_re * z_im + x_re * y_im * z_re + x_im * y_re * z_re \
        - x_im * y_im * z_im
    if split:
        return res_re, res_im
    else:
        return res_re + res_im


def compute_error(u, uh, relative=True):
    v = u - uh
    err = fd.assemble(fd.inner(v, v) * fd.dx(1))**0.5
    if relative:
        err /= fd.assemble(fd.inner(u, u) * fd.dx(1))**0.5
    return err


def far_field(k, u, theta):
    mesh = u.function_space().mesh()
    n = fd.FacetNormal(mesh)
    y = fd.SpatialCoordinate(mesh)

    x = fd.Constant((np.cos(theta), np.sin(theta)))
    phi = fd.pi / 4 - k * fd.inner(x, y)
    f = (fd.cos(phi), fd.sin(phi))
    g = (fd.inner(k * u[1] * x - fd.grad(u[0]), -n),
         fd.inner(-k * u[0] * x - fd.grad(u[1]), -n))
    h = prod2(f, g, split=True)
    res_re = 1 / np.sqrt(8*np.pi*k) * fd.assemble(h[0] * fd.ds(1))
    res_im = 1 / np.sqrt(8*np.pi*k) * fd.assemble(h[1] * fd.ds(1))
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


def plot_far_field(k, u):
    theta = np.linspace(0, 2 * np.pi, 100)
    u_inf = []
    for t in list(theta):
        u_inf.append(far_field(k, u, t))
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
