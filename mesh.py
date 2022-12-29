import gmsh
import numpy as np
import firedrake as fd


def generate_mesh(a0, a1, b0, b1, shape, h0, level=0):
    gmsh.initialize()

    # Absorbing layer
    gmsh.model.occ.addPoint(a0, -b0, 0, 1, 1)
    gmsh.model.occ.addPoint(a0, b0, 0, 1, 2)
    gmsh.model.occ.addPoint(-a0, b0, 0, 1, 3)
    gmsh.model.occ.addPoint(-a0, -b0, 0, 1, 4)
    gmsh.model.occ.addPoint(a1, -b1, 0, 1, 5)
    gmsh.model.occ.addPoint(a1, b1, 0, 1, 6)
    gmsh.model.occ.addPoint(-a1, b1, 0, 1, 7)
    gmsh.model.occ.addPoint(-a1, -b1, 0, 1, 8)
    kp_start = 9

    gmsh.model.occ.addLine(1, 2, 1)
    gmsh.model.occ.addLine(2, 3, 2)
    gmsh.model.occ.addLine(3, 4, 3)
    gmsh.model.occ.addLine(4, 1, 4)
    gmsh.model.occ.addLine(5, 6, 5)
    gmsh.model.occ.addLine(6, 7, 6)
    gmsh.model.occ.addLine(7, 8, 7)
    gmsh.model.occ.addLine(8, 5, 8)
    kc_start = 9

    # Obstacle
    if shape == "sphere":
        gmsh.model.occ.addCircle(0, 0, 0, 1, kc_start)
        kc_end = kc_start + 1

    elif shape == "kite":
        ts = np.linspace(0, 2 * np.pi, 100 * 2**level, endpoint=False)
        kp_start = 17
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

    else:
        print("Unsupported shape.")
        raise NotImplementedError

    gmsh.model.occ.addCurveLoop(range(kc_start, kc_end), 1)
    gmsh.model.occ.addCurveLoop(range(1, 5), 2)
    gmsh.model.occ.addCurveLoop(range(5, 9), 3)

    gmsh.model.occ.addPlaneSurface([1, 2], 1)
    gmsh.model.occ.addPlaneSurface([2, 3], 2)

    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(1, range(kc_start, kc_end), 1, name="Gamma")
    gmsh.model.addPhysicalGroup(1, range(1, 5), 2, name="Gamma_I")
    gmsh.model.addPhysicalGroup(1, range(5, 9), 3, name="Gamma_D")

    gmsh.model.addPhysicalGroup(2, [1], 1, name="Omega_F")
    gmsh.model.addPhysicalGroup(2, [2], 2, name="Omega_A")

    gmsh.option.setNumber("Mesh.MeshSizeFactor", h0 / 2**level)
    gmsh.model.mesh.generate(2)

    msh_file = shape + str(level) + ".msh"
    gmsh.write(msh_file)

    gmsh.finalize()

    return fd.Mesh(msh_file)
