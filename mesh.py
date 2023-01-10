import gmsh
import numpy as np
import firedrake as fd


def generate_mesh(a0, a1, b0, b1, shape, h0, level=0, R0=None, R1=None):
    gmsh.initialize()

    # Absorbing layer
    gmsh.model.occ.addPoint(a0, -b0, 0, 1, 1)
    gmsh.model.occ.addPoint(a0, b0, 0, 1, 2)
    gmsh.model.occ.addPoint(-a0, b0, 0, 1, 3)
    gmsh.model.occ.addPoint(-a0, -b0, 0, 1, 4)
    gmsh.model.occ.addPoint(a1, -b1, 0, 1, 5)
    gmsh.model.occ.addPoint(a1, -b0, 0, 1, 6)
    gmsh.model.occ.addPoint(a1, b0, 0, 1, 7)
    gmsh.model.occ.addPoint(a1, b1, 0, 1, 8)
    gmsh.model.occ.addPoint(a0, b1, 0, 1, 9)
    gmsh.model.occ.addPoint(-a0, b1, 0, 1, 10)
    gmsh.model.occ.addPoint(-a1, b1, 0, 1, 11)
    gmsh.model.occ.addPoint(-a1, b0, 0, 1, 12)
    gmsh.model.occ.addPoint(-a1, -b0, 0, 1, 13)
    gmsh.model.occ.addPoint(-a1, -b1, 0, 1, 14)
    gmsh.model.occ.addPoint(-a0, -b1, 0, 1, 15)
    gmsh.model.occ.addPoint(a0, -b1, 0, 1, 16)
    kp_start = 17

    gmsh.model.occ.addLine(1, 2, 1)
    gmsh.model.occ.addLine(2, 3, 2)
    gmsh.model.occ.addLine(3, 4, 3)
    gmsh.model.occ.addLine(4, 1, 4)
    gmsh.model.occ.addLine(6, 7, 5)
    gmsh.model.occ.addLine(9, 10, 6)
    gmsh.model.occ.addLine(12, 13, 7)
    gmsh.model.occ.addLine(15, 16, 8)
    gmsh.model.occ.addLine(2, 7, 9)
    gmsh.model.occ.addLine(7, 8, 10)
    gmsh.model.occ.addLine(8, 9, 11)
    gmsh.model.occ.addLine(9, 2, 12)
    gmsh.model.occ.addLine(3, 10, 13)
    gmsh.model.occ.addLine(10, 11, 14)
    gmsh.model.occ.addLine(11, 12, 15)
    gmsh.model.occ.addLine(12, 3, 16)
    gmsh.model.occ.addLine(4, 13, 17)
    gmsh.model.occ.addLine(13, 14, 18)
    gmsh.model.occ.addLine(14, 15, 19)
    gmsh.model.occ.addLine(15, 4, 20)
    gmsh.model.occ.addLine(1, 16, 21)
    gmsh.model.occ.addLine(16, 5, 22)
    gmsh.model.occ.addLine(5, 6, 23)
    gmsh.model.occ.addLine(6, 1, 24)
    kc_start = 25

    # Obstacle
    if shape == "sphere":
        gmsh.model.occ.addCircle(0.1, 0.1, 0, 1, kc_start)
        kc_end = kc_start + 1

    elif shape == "kite":
        ts = np.linspace(0, 2 * np.pi, 100 * 2**level, endpoint=False)
        kp = kp_start
        for t in ts:
            x = np.cos(t) + 0.65 * np.cos(2 * t) - 0.65
            y = 1.5 * np.sin(t)
            gmsh.model.occ.addPoint(x, y, 0, 1, kp)
            kp += 1
        kp_end = kp

        kc = kc_start
        for kp in range(kp_start, kp_end - 1):
            gmsh.model.occ.addLine(kp, kp + 1, kc)
            kc += 1
        gmsh.model.occ.addLine(kp_end - 1, kp_start, kc)
        kc_end = kc + 1

    elif shape == "square":
        gmsh.model.occ.addPoint(1, -1, 0, 1, kp_start)
        gmsh.model.occ.addPoint(1, 1, 0, 1, kp_start + 1)
        gmsh.model.occ.addPoint(-1, 1, 0, 1, kp_start + 2)
        gmsh.model.occ.addPoint(-1, -1, 0, 1, kp_start + 3)

        kc = kc_start
        for kp in range(kp_start, kp_start + 3):
            gmsh.model.occ.addLine(kp, kp + 1, kc)
            kc += 1
        gmsh.model.occ.addLine(kp_start + 3, kp_start, kc)
        kc_end = kc + 1

    else:
        print("Unsupported shape.")
        raise NotImplementedError

    gmsh.model.occ.addCurveLoop(range(1, 5), 1)
    gmsh.model.occ.addCurveLoop([5, -9, -1, -24], 2)
    gmsh.model.occ.addCurveLoop([-12, 6, -13, -2], 3)
    gmsh.model.occ.addCurveLoop([-3, -16, 7, -17], 4)
    gmsh.model.occ.addCurveLoop([-21, -4, -20, 8], 5)
    gmsh.model.occ.addCurveLoop(range(9, 13), 6)
    gmsh.model.occ.addCurveLoop(range(13, 17), 7)
    gmsh.model.occ.addCurveLoop(range(17, 21), 8)
    gmsh.model.occ.addCurveLoop(range(21, 25), 9)
    gmsh.model.occ.addCurveLoop(range(kc_start, kc_end), 10)

    for cl in range(1, 9):
        gmsh.model.occ.addPlaneSurface([cl + 1], cl)

    if R0 is None:
        gmsh.model.occ.addPlaneSurface([1, 10], 9)
    else:
        circ0 = gmsh.model.occ.addCircle(0, 0, 0, R0)
        cl0 = gmsh.model.occ.addCurveLoop([circ0])
        gmsh.model.occ.addPlaneSurface([10, cl0], 9)
        if R1 is None:
            gmsh.model.occ.addPlaneSurface([cl0, 1], 10)
        else:
            circ1 = gmsh.model.occ.addCircle(0, 0, 0, R1)
            cl1 = gmsh.model.occ.addCurveLoop([circ1])
            gmsh.model.occ.addPlaneSurface([cl0, cl1], 10)
            gmsh.model.occ.addPlaneSurface([cl1, 1], 11)

    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(1, range(kc_start, kc_end), 1, name="Gamma")
    gmsh.model.addPhysicalGroup(1, range(1, 5), 2, name="Gamma_I")
    gmsh.model.addPhysicalGroup(
        1, list(range(5, 9)) + [10, 11, 14, 15, 18, 19, 22, 23], 3,
        name="Gamma_D")
    if R0 is not None:
        gmsh.model.addPhysicalGroup(1, [circ0], 4, name="R0")
    if R1 is not None:
        gmsh.model.addPhysicalGroup(1, [circ1], 5, name="R1")

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

    gmsh.option.setNumber("Mesh.MeshSizeFactor", h0 / 2**level)
    gmsh.model.mesh.generate(2)

    msh_file = shape + str(level) + ".msh"
    gmsh.write(msh_file)

    gmsh.finalize()

    return fd.Mesh(msh_file)
