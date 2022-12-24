import firedrake as fd
import numpy as np
from scipy.special import hankel1
import matplotlib.pyplot as plt


def assign_data(v, v_data):
    v.sub(0).dat.data[:] = v_data.real
    v.sub(1).dat.data[:] = v_data.imag


def prod2(x, y):
    x_re, x_im = x
    y_re, y_im = y
    res_re = x_re * y_re - x_im * y_im
    res_im = x_re * y_im + x_im * y_re
    return res_re + res_im


def prod3(x, y, z):
    x_re, x_im = x[0], x[1]
    y_re, y_im = y[0], y[1]
    z_re, z_im = z[0], z[1]
    res_re = x_re * y_re * z_re - x_re * y_im * z_im - x_im * y_re * z_im \
        - x_im * y_im * z_re
    res_im = x_re * y_re * z_im + x_re * y_im * z_re + x_im * y_re * z_re \
        - x_im * y_im * z_im
    return res_re + res_im


def compute_error(u, uh, relative=True):
    v = u - uh
    err = fd.assemble(fd.inner(v, v) * fd.dx(1))**0.5
    if relative:
        err /= fd.assemble(fd.inner(u, u) * fd.dx(1))**0.5
    return err


c = 340
a0 = b0 = 2.0
a1 = b1 = 2.25
x0 = np.array([0.5, 0])
h0 = 2 * a0 / 16

neumann = False
max_level = 6
levels = np.arange(max_level)
mesh_hierarchy = fd.MeshHierarchy(fd.Mesh("mesh.msh"), max_level)
hs = h0 / 2**levels

print("----------------------------------------")
for omega in [250, 750, 1250]:
    print(f"angular freqency = {omega}")
    k = omega / c
    errors = []
    for level in levels:
        mesh = mesh_hierarchy[level]
        W = fd.VectorFunctionSpace(mesh, "CG", 1)
        p = fd.TrialFunction(W)
        q = fd.TestFunction(W)
        s = fd.Function(W)
        exact = fd.Function(W)

        # Exact solution
        X_data = fd.interpolate(mesh.coordinates, W).dat.data_ro
        dist = np.linalg.norm(X_data - x0, axis=1)
        exact_data = 1j / 4 * hankel1(0, k * dist)
        assign_data(exact, exact_data)

        # Neumann data
        if neumann:
            g = fd.Function(W)
            g_data = -1j / 4 * hankel1(1, k * dist) \
                * k / dist * np.einsum("ij,ij->i", X_data - x0, -X_data)
            assign_data(g, g_data)

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

        # Variational form
        px, py = p.dx(0), p.dx(1)
        qx, qy = q.dx(0), q.dx(1)
        a = (prod2(px, qx) + prod2(py, qy) - k**2 * prod2(p, q)) * fd.dx(1) \
            + (prod3(c1x, px, qx) + prod3(c2x, py, qy)
                - k**2 * prod3(c3x, p, q)) * fd.dx(2) \
            + (prod3(c1y, px, qx) + prod3(c2y, py, qy)
                - k**2 * prod3(c3y, p, q)) * fd.dx(3) \
            + (prod3(c1xy, px, qx) + prod3(c2xy, py, qy)
                - k**2 * prod3(c3xy, p, q)) * fd.dx(4)
        if neumann:
            L = (prod2(g, q)) * fd.ds(1)
            bcs = [fd.DirichletBC(W, (0., 0), 3)]
        else:
            L = fd.Constant(0) * q[0] * fd.dx
            bcs = [fd.DirichletBC(W, exact, 1), fd.DirichletBC(W, (0., 0.), 3)]

        # Solution
        fd.solve(a == L, s, bcs=bcs)

        rel_err = compute_error(exact, s)
        print(f"refinement level {level}, relative error {rel_err:.2%}")
        errors.append(rel_err)
    plt.loglog(hs, errors, "-o",
               label=r"Relative error of $\omega=$"+f"{omega}")
    print("----------------------------------------")

plt.loglog(hs, hs**2, "k", label=r"Order $h^2$")
plt.legend()
plt.xlabel(r"Mesh width $h$")
plt.ylabel("Relative error")
if neumann:
    plt.title("Neumann boundary condition")
else:
    plt.title("Dirichlet boundary condition")
plt.tight_layout()
plt.show()
