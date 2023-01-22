from math import ceil
import gmsh
import numpy as np
import firedrake as fd


def generate_mesh(a0, a1, b0, b1, shape, N, level=0, R0=None, R1=None,
                  shift=(0, 0), suffix=""):
    h0 = min(a1 - a0, b1 - b0) / N
    x0, y0 = shift

    gmsh.initialize()

    # Absorbing layer
    gmsh.model.geo.addPoint(a0, -b0, 0, 1, 1)
    gmsh.model.geo.addPoint(a0, b0, 0, 1, 2)
    gmsh.model.geo.addPoint(-a0, b0, 0, 1, 3)
    gmsh.model.geo.addPoint(-a0, -b0, 0, 1, 4)
    gmsh.model.geo.addPoint(a1, -b1, 0, 1, 5)
    gmsh.model.geo.addPoint(a1, -b0, 0, 1, 6)
    gmsh.model.geo.addPoint(a1, b0, 0, 1, 7)
    gmsh.model.geo.addPoint(a1, b1, 0, 1, 8)
    gmsh.model.geo.addPoint(a0, b1, 0, 1, 9)
    gmsh.model.geo.addPoint(-a0, b1, 0, 1, 10)
    gmsh.model.geo.addPoint(-a1, b1, 0, 1, 11)
    gmsh.model.geo.addPoint(-a1, b0, 0, 1, 12)
    gmsh.model.geo.addPoint(-a1, -b0, 0, 1, 13)
    gmsh.model.geo.addPoint(-a1, -b1, 0, 1, 14)
    gmsh.model.geo.addPoint(-a0, -b1, 0, 1, 15)
    gmsh.model.geo.addPoint(a0, -b1, 0, 1, 16)
    p_start = 17

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    gmsh.model.geo.addLine(6, 7, 5)
    gmsh.model.geo.addLine(9, 10, 6)
    gmsh.model.geo.addLine(12, 13, 7)
    gmsh.model.geo.addLine(15, 16, 8)
    gmsh.model.geo.addLine(2, 7, 9)
    gmsh.model.geo.addLine(7, 8, 10)
    gmsh.model.geo.addLine(8, 9, 11)
    gmsh.model.geo.addLine(9, 2, 12)
    gmsh.model.geo.addLine(3, 10, 13)
    gmsh.model.geo.addLine(10, 11, 14)
    gmsh.model.geo.addLine(11, 12, 15)
    gmsh.model.geo.addLine(12, 3, 16)
    gmsh.model.geo.addLine(4, 13, 17)
    gmsh.model.geo.addLine(13, 14, 18)
    gmsh.model.geo.addLine(14, 15, 19)
    gmsh.model.geo.addLine(15, 4, 20)
    gmsh.model.geo.addLine(1, 16, 21)
    gmsh.model.geo.addLine(16, 5, 22)
    gmsh.model.geo.addLine(5, 6, 23)
    gmsh.model.geo.addLine(6, 1, 24)
    c_start = 25

    # Obstacle
    if shape == "sphere":
        r = 1.
        p0 = gmsh.model.geo.addPoint(x0, y0, 0, 1)
        p1 = gmsh.model.geo.addPoint(x0 + r, y0, 0, 1)
        p2 = gmsh.model.geo.addPoint(x0, y0 + r, 0, 1)
        p3 = gmsh.model.geo.addPoint(x0 - r, y0, 0, 1)
        p4 = gmsh.model.geo.addPoint(x0, y0 - r, 0, 1)
        gmsh.model.geo.addCircleArc(p1, p0, p2, c_start)
        gmsh.model.geo.addCircleArc(p2, p0, p3, c_start + 1)
        gmsh.model.geo.addCircleArc(p3, p0, p4, c_start + 2)
        gmsh.model.geo.addCircleArc(p4, p0, p1, c_start + 3)
        c_end = c_start + 4

    elif shape == "kite":
        ts = np.linspace(0, 2 * np.pi, 100 * 2**level, endpoint=False)
        p = p_start
        for t in ts:
            x = x0 + np.cos(t) + 0.65 * np.cos(2 * t) - 0.65
            y = y0 + 1.5 * np.sin(t)
            gmsh.model.geo.addPoint(x, y, 0, 1, p)
            p += 1
        p_end = p

        c = c_start
        for p in range(p_start, p_end - 1):
            gmsh.model.geo.addLine(p, p + 1, c)
            c += 1
        gmsh.model.geo.addLine(p_end - 1, p_start, c)
        c_end = c + 1

    elif shape == "square":
        gmsh.model.geo.addPoint(x0+1, y0-1, 0, 1, p_start)
        gmsh.model.geo.addPoint(x0+1, y0+1, 0, 1, p_start + 1)
        gmsh.model.geo.addPoint(x0-1, y0+1, 0, 1, p_start + 2)
        gmsh.model.geo.addPoint(x0-1, y0-1, 0, 1, p_start + 3)

        c = c_start
        for p in range(p_start, p_start + 3):
            gmsh.model.geo.addLine(p, p + 1, c)
            c += 1
        gmsh.model.geo.addLine(p_start + 3, p_start, c)
        c_end = c + 1

    else:
        print("Unsupported shape.")
        raise NotImplementedError

    gmsh.model.geo.addCurveLoop(range(1, 5), 1)
    gmsh.model.geo.addCurveLoop([5, -9, -1, -24], 2)
    gmsh.model.geo.addCurveLoop([-12, 6, -13, -2], 3)
    gmsh.model.geo.addCurveLoop([-3, -16, 7, -17], 4)
    gmsh.model.geo.addCurveLoop([-21, -4, -20, 8], 5)
    gmsh.model.geo.addCurveLoop(range(9, 13), 6)
    gmsh.model.geo.addCurveLoop(range(13, 17), 7)
    gmsh.model.geo.addCurveLoop(range(17, 21), 8)
    gmsh.model.geo.addCurveLoop(range(21, 25), 9)
    gmsh.model.geo.addCurveLoop(range(c_start, c_end), 10)

    for cl in range(1, 9):
        gmsh.model.geo.addPlaneSurface([cl + 1], cl)

    if R0 is None:
        gmsh.model.geo.addPlaneSurface([1, 10], 9)
    else:
        p0 = gmsh.model.geo.addPoint(0, 0, 0, 1)
        p1 = gmsh.model.geo.addPoint(R0, 0, 0, 1)
        p2 = gmsh.model.geo.addPoint(0, R0, 0, 1)
        p3 = gmsh.model.geo.addPoint(-R0, 0, 0, 1)
        p4 = gmsh.model.geo.addPoint(0, -R0, 0, 1)
        c_R0 = []
        c_R0.append(gmsh.model.geo.addCircleArc(p1, p0, p2))
        c_R0.append(gmsh.model.geo.addCircleArc(p2, p0, p3))
        c_R0.append(gmsh.model.geo.addCircleArc(p3, p0, p4))
        c_R0.append(gmsh.model.geo.addCircleArc(p4, p0, p1))
        cl0 = gmsh.model.geo.addCurveLoop(c_R0)
        gmsh.model.geo.addPlaneSurface([10, cl0], 9)
        if R1 is None:
            gmsh.model.geo.addPlaneSurface([cl0, 1], 10)
        else:
            p0 = gmsh.model.geo.addPoint(0, 0, 0, 1)
            p1 = gmsh.model.geo.addPoint(R1, 0, 0, 1)
            p2 = gmsh.model.geo.addPoint(0, R1, 0, 1)
            p3 = gmsh.model.geo.addPoint(-R1, 0, 0, 1)
            p4 = gmsh.model.geo.addPoint(0, -R1, 0, 1)
            c_R1 = []
            c_R1.append(gmsh.model.geo.addCircleArc(p1, p0, p2))
            c_R1.append(gmsh.model.geo.addCircleArc(p2, p0, p3))
            c_R1.append(gmsh.model.geo.addCircleArc(p3, p0, p4))
            c_R1.append(gmsh.model.geo.addCircleArc(p4, p0, p1))
            cl1 = gmsh.model.geo.addCurveLoop(c_R1)
            gmsh.model.geo.addPlaneSurface([cl0, cl1], 10)
            gmsh.model.geo.addPlaneSurface([cl1, 1], 11)

    NN = ceil(2 * max(a0, b0) / min(a1 - a0, b1 - b0)) * N
    for c in range(1, 9):
        gmsh.model.geo.mesh.setTransfiniteCurve(c, NN + 1)

    for s in range(1, 5):
        gmsh.model.geo.mesh.setTransfiniteSurface(s)

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, range(c_start, c_end), 1, name="Gamma")
    gmsh.model.addPhysicalGroup(1, range(1, 5), 2, name="Gamma_I")
    gmsh.model.addPhysicalGroup(
        1, list(range(5, 9)) + [10, 11, 14, 15, 18, 19, 22, 23], 3,
        name="Gamma_D")
    if R0 is not None:
        gmsh.model.addPhysicalGroup(1, c_R0, 4, name="R0")
    if R1 is not None:
        gmsh.model.addPhysicalGroup(1, c_R1, 5, name="R1")

    if R0 is None:
        gmsh.model.addPhysicalGroup(2, [9], 1, name="Omega_F")
    elif R1 is None:
        gmsh.model.addPhysicalGroup(2, [9, 10], 1, name="Omega_F")
    else:
        gmsh.model.addPhysicalGroup(2, [9, 11], 1, name="Omega_F")
        gmsh.model.addPhysicalGroup(2, [10], 5, name="Omega_inf")

    gmsh.model.addPhysicalGroup(2, [1, 3], 2, name="Omega_A_x")
    gmsh.model.addPhysicalGroup(2, [2, 4], 3, name="Omega_A_y")
    gmsh.model.addPhysicalGroup(2, range(5, 9), 4, name="Omega_A_xy")

    gmsh.option.setNumber("Mesh.MeshSizeFactor", h0)
    gmsh.model.mesh.generate(2)
    for _ in range(level):
        gmsh.model.mesh.refine()

    msh_file = shape + str(level) + suffix + ".msh"
    gmsh.write(msh_file)

    gmsh.finalize()

    return fd.Mesh(msh_file)
