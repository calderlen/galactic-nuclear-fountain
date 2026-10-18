#!/usr/bin/env python3
"""Equation-annotated nuclear fountain diagram; export a single vector PDF.

The paths are illustrative. Equations follow model/orbits.cpp,
model/launch_distribution.cpp, model/kernel_runner.cpp,
model/landing_kernel.cpp, and model/models.cpp.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath


INK = "#27343A"
MUTED = "#657279"
ORANGE = "#CC5C27"
BLUE = "#327990"
PALE_BLUE = "#BCDCE5"
HALO = "#F2F7F9"
EQUATION_FONT_SIZE = 15


def trajectory(t, control):
    """Quadratic Bezier shared by paths, parcel markers, and direction arrows."""
    t = np.asarray(t)[..., None]
    p0, p1, p2 = np.asarray(control)
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def arrow(ax, start, end, color, lw=1.3, size=10, rad=0, zorder=5):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=size,
        connectionstyle=f"arc3,rad={rad}", color=color, linewidth=lw,
        shrinkA=0, shrinkB=0, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def orbit(ax, control, color, lw, arrows=(0.24, 0.72), zorder=4):
    points = trajectory(np.linspace(0, 1, 400), control)
    ax.plot(*points.T, color=color, lw=lw, solid_capstyle="round", zorder=zorder)
    for t in arrows:
        arrow(ax, trajectory(t - 0.016, control), trajectory(t + 0.016, control),
              color, lw=lw, size=11 if lw > 2 else 8, zorder=zorder + 1)


def label(ax, x, y, text, *, size=14, color=INK, ha="left"):
    return ax.text(x, y, text, fontsize=size, color=color, ha=ha, va="center", zorder=20)


def leader(ax, text, xy, xytext, *, ha="left", color=INK, size=11):
    return ax.annotate(
        text, xy=xy, xytext=xytext, ha=ha, va="center", fontsize=size,
        color=color,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=5, shrinkB=4),
        zorder=20,
    )


def dimension(ax, x0, x1, y, text=None):
    ax.plot([x0, x1], [y, y], lw=0.75, color=MUTED, zorder=10)
    for x in (x0, x1):
        ax.plot([x, x], [y - 0.06, y + 0.06], lw=0.75, color=MUTED, zorder=10)
    if text:
        label(ax, (x0 + x1) / 2, y - 0.25, text, ha="center", size=EQUATION_FONT_SIZE)


def draw_scene(ax):
    # All local annotations occupy the same R-z drawing, without inset panels.
    dome = MplPath(
        [(-8.7, 0.03), (-9.0, 10.6), (8.6, 10.7), (8.7, 0.03), (-8.7, 0.03)],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY],
    )
    ax.add_patch(PathPatch(dome, facecolor=HALO, edgecolor="none", zorder=0))
    label(ax, 2.65, 8.0, "Rotating CGM", color=BLUE, size=12.5, ha="center")
    label(ax, 2.65, 7.48,
          r"$\mathbf{v}_{\rm CGM}=\beta v_c(R)\,\hat{\mathbf{e}}_\phi,"
          r"\qquad \rho_{\rm CGM}=\mathrm{const.}$", size=EQUATION_FONT_SIZE, color=BLUE, ha="center")
    arrow(ax, (-1.5, 6.91), (1.2, 6.98), BLUE, lw=1.1, rad=-0.10)
    arrow(ax, (6.40, 6.98), (8.55, 6.82), BLUE, lw=1.1, rad=-0.10)

    # Illustrative launch directions, not a statistical sample of the PDFs.
    # A Bezier's initial tangent is p1-p0. Set its projected slope from the
    # launch components, rather than making every launch nearly vertical.
    # The left side is mirrored: increasing cylindrical R points leftward.
    launch_point = np.array((-0.85, 0.15))
    launch_examples = [
        # theta [deg], psi [deg], control height, landing x, color
        (10.0,   0.0, 8.35, -3.9, "#CF956C"),
        (26.0,  35.0, 6.20, -4.6, "#D9A782"),
        (43.0,   0.0, 3.70, -5.0, "#E1B795"),
        (60.0, -20.0, 1.90, -5.4, "#E8C6A9"),
    ]
    left_paths = []
    for theta_deg, psi_deg, height, landing_x, color in launch_examples:
        slope = np.tan(np.deg2rad(theta_deg)) * np.cos(np.deg2rad(psi_deg))
        control = launch_point + np.array((-height * slope, height))
        path = (launch_point, control, (landing_x, launch_point[1]))
        left_paths.append(path)
        orbit(ax, path, color, lw=1.35, arrows=(0.075, 0.25, 0.79))
    main = ((0.85, 0.15), (1.35, 11.3), (7.8, 0.15))
    orbit(ax, main, ORANGE, lw=2.6, arrows=(0.20, 0.70, 0.86), zorder=7)
    label(ax, -4.15, 5.44, "Projected launch directions", color=MUTED, size=11.5)
    leader(ax, r"$v_{R,0}/v_{z,0}=\tan\theta\cos\psi$",
           trajectory(0.46, left_paths[0]), (-4.15, 4.95),
           color=MUTED, size=EQUATION_FONT_SIZE)

    # Resolve the local interaction at the apex, where v_z=0. The orange
    # arrow and blue drag arrow are R-z projections, not azimuthal vectors.
    # Incoming blue streams depict capture in the parcel frame. Their
    # curvature is illustrative; no resolved wake or cooling layer is assumed.
    growth = trajectory(0.5, main)
    ax.add_patch(Ellipse(growth, 0.64, 0.64, facecolor=PALE_BLUE, edgecolor=BLUE, lw=0.75, zorder=9))
    ax.add_patch(Ellipse(growth, 0.43, 0.43, facecolor=ORANGE, edgecolor="white", lw=1.0, zorder=10))
    gas = growth + np.array([[0.90, 0.52], [1.24, 0.48], [1.60, 0.54], [1.00, 0.20], [1.40, 0.18], [1.75, 0.24]])
    ax.scatter(*gas.T, s=[14, 19, 11, 12, 16, 10], c=BLUE, alpha=0.60, lw=0, zorder=9)
    arrow(ax, growth + (1.12, 0.47), growth + (0.23, 0.23), BLUE, lw=1.05, size=8, rad=0.15, zorder=11)
    arrow(ax, growth + (1.43, 0.16), growth + (0.32, 0.02), BLUE, lw=1.05, size=8, rad=-0.06, zorder=11)
    label(ax, 3.82, 6.88, "Mass and momentum input", size=10.5, color=BLUE)
    label(ax, 3.82, 6.52, r"$\dot m,\quad \dot m\,\mathbf{v}_{\rm CGM}$", size=EQUATION_FONT_SIZE, color=BLUE)

    arrow(ax, growth + (-0.35, 0.97), growth + (0.72, 0.97), ORANGE, lw=1.4, size=9, zorder=11)
    label(ax, growth[0] + 0.16, growth[1] + 1.23, r"$v_R$", size=EQUATION_FONT_SIZE, color=ORANGE, ha="center")
    arrow(ax, growth + (-0.38, 0.0), growth + (-1.45, 0.0), BLUE, lw=1.5, size=10, zorder=11)
    label(ax, growth[0] - 1.45, growth[1] + 0.29, r"$-\Gamma v_R$", size=EQUATION_FONT_SIZE, color=BLUE)
    leader(ax, r"$m(t),\ j_z(t)$", growth + (-0.18, 0.26), (0.32, 6.42), size=EQUATION_FONT_SIZE)
    arrow(ax, growth + (-0.13, -0.37), growth + (-0.54, -1.03), MUTED, lw=1.1, size=9, zorder=11)
    label(ax, growth[0] + 0.10, growth[1] - 0.80, r"$-\nabla\Phi$", size=EQUATION_FONT_SIZE, color=MUTED)

    # Galactic disk, nuclear ring, and radial transport after deposition.
    ax.add_patch(Ellipse((0, 0), 17.3, 0.45, facecolor="#E3E4DF", edgecolor="#929B98", lw=0.95, zorder=10))
    ax.add_patch(Ellipse((0, 0.12), 2.2, 0.75, facecolor="#CDD2CB", edgecolor="none", zorder=11))
    ax.add_patch(Ellipse((0, 0.15), 1.7, 0.32, facecolor="#F9E0CD", edgecolor=ORANGE, lw=1.8, zorder=12))
    ax.add_patch(Ellipse((0, 0.15), 1.3, 0.18, facecolor="#F7F6F0", edgecolor="none", zorder=13))
    ax.scatter([0.85], [0.15], s=35, c=ORANGE, edgecolors="white", lw=0.8, zorder=15)
    ax.scatter([7.8], [0.15], s=180, c=ORANGE, edgecolors="white", lw=1.1, zorder=15)
    label(ax, 1.20, -0.04, r"$m_0,\ t_0$", size=EQUATION_FONT_SIZE)
    leader(ax, r"$m_{\rm land},\ j_{\rm land}$", (7.8, 0.15), (8.58, 0.46), size=EQUATION_FONT_SIZE)
    # Attach callouts to the actual outlines, with separate routes for labels
    # and dimensions. The ring callout stays left of the radial guides.
    ring_x = -0.55
    ring_y = 0.15 - 0.16 * np.sqrt(1 - (ring_x / 0.85) ** 2)
    ax.plot([ring_x, ring_x, -1.42], [ring_y - 0.02, -0.76, -0.76],
            color=MUTED, lw=0.8, solid_capstyle="round", zorder=16)
    label(ax, -1.58, -0.76, "Circumnuclear ring", ha="right", size=11)
    disk_x = -7.9
    disk_y = -0.225 * np.sqrt(1 - (disk_x / 8.65) ** 2)
    ax.plot([disk_x, disk_x], [disk_y - 0.02, -0.40], color=MUTED, lw=0.8, solid_capstyle="round", zorder=16)
    label(ax, disk_x, -0.66, "Galactic disk", ha="center", size=11)
    arrow(ax, (6.65, 0.0), (4.2, 0.0), BLUE, lw=1.2, size=9, zorder=15)
    label(ax, 5.45, -0.47,
          r"$j_{\rm land}<j_{\rm disk}\ \Rightarrow\ v_R<0$",
          size=EQUATION_FONT_SIZE, color=BLUE, ha="center")

    # Cylindrical coordinates and radii from the common galactic center.
    # Dimension extensions start below the disk rather than passing through
    # the ring. R_nuc is centered under its bracket, with no crossing leader.
    ax.plot([0, 0], [-0.30, -1.16], color="#A6B1B5", lw=0.65, zorder=9)
    arrow(ax, (0, 0.70), (0, 2.2), MUTED, lw=0.75, size=8)
    label(ax, -0.10, 2.38, r"$z$", size=EQUATION_FONT_SIZE)
    ax.plot([0.85, 0.85], [-0.30, -0.44], color="#A6B1B5", lw=0.65, zorder=9)
    dimension(ax, 0, 0.85, -0.44, r"$R_{\rm nuc}$")
    ax.plot([7.8, 7.8], [-0.30, -1.16], color="#A6B1B5", lw=0.65, zorder=9)
    dimension(ax, 0, 7.8, -1.10, r"$R_{\rm land}$")


def annotate_physics(ax):
    # Axisymmetric orbit: exact cylindrical equations from model/orbits.cpp.
    label(ax, -8.9, 7.78, "Orbit in the galactic potential", size=11.5, color=MUTED)
    label(ax, -8.9, 7.22, r"$\Phi=\Phi_{\rm MN}+\Phi_{\rm H}+\Phi_{\rm NFW}$", size=EQUATION_FONT_SIZE)
    label(ax, -8.9, 6.58, r"$\dot R=v_R,\quad \dot z=v_z,\quad v_c^2=R\,\partial_R\Phi(R,0)$", size=EQUATION_FONT_SIZE)
    label(ax, -8.9, 5.90, r"$\dot v_R=-\partial_R\Phi+\dfrac{j_z^2}{R^3}-\Gamma v_R$", size=EQUATION_FONT_SIZE)
    label(ax, -8.9, 5.20, r"$\dot v_z=-\partial_z\Phi-\Gamma v_z$", size=EQUATION_FONT_SIZE)

    # Probability densities from launch_distribution.cpp, for nonzero widths.
    label(ax, -8.9, 3.88, "Launch distribution", size=11.5, color=ORANGE)
    label(ax, -8.9, 3.30, r"$p(R_0)\propto R_0\,e^{-(R_0-R_{\rm nuc})^2/(2\sigma_R^2)}$", size=EQUATION_FONT_SIZE)
    label(ax, -8.9, 2.64, r"$p(v_k)\propto v_k^2\,e^{-v_k^2/(2h_v^2)}$", size=EQUATION_FONT_SIZE)
    label(ax, -8.9, 1.97, r"$p(\theta)\propto\sin\theta\,e^{-\theta^2/(2\sigma_\theta^2)}$", size=EQUATION_FONT_SIZE)
    label(ax, -8.9, 1.31, r"$0\leq\theta<\pi/2,\qquad \psi\sim\mathcal{U}(0,2\pi)$", size=EQUATION_FONT_SIZE)

    # Separate the existing coupling into accreted-momentum loading and drag;
    # substituting dot(m) recovers (epsilon+C_D/2)*rho_CGM*sigma*u/m.
    label(ax, 5.75, 6.31, "Condensation and drag", size=11.5, color=BLUE)
    label(ax, 5.75, 5.80, r"$\dot m=\epsilon\,\rho_{\rm CGM}\sigma u$", size=EQUATION_FONT_SIZE)
    label(ax, 5.75, 5.12,
          r"$\Gamma=\dfrac{\dot m}{m}+\dfrac{C_D}{2}"
          r"\dfrac{\rho_{\rm CGM}\sigma u}{m}$", size=EQUATION_FONT_SIZE)
    label(ax, 5.75, 4.47,
          r"$u=|\mathbf{v}-\mathbf{v}_{\rm CGM}|,\qquad \sigma=\mathrm{const.}$",
          size=EQUATION_FONT_SIZE)

    label(ax, 1.55, 4.13, "Angular-momentum exchange", size=11, color=BLUE)
    label(ax, 1.55, 3.60, r"$\dot j_z=\Gamma\left(\beta Rv_c-j_z\right)$", size=EQUATION_FONT_SIZE)
    label(ax, 1.55, 3.08, r"$j_z=Rv_\phi$", size=EQUATION_FONT_SIZE)

    # Initial conditions include the azimuthal kick, not just circular rotation.
    label(ax, 1.75, 2.49, "At launch", size=10.5, color=ORANGE)
    label(ax, 1.75, 2.02, r"$v_{R,0}=v_k\sin\theta\cos\psi$", size=EQUATION_FONT_SIZE)
    label(ax, 1.75, 1.53, r"$v_{z,0}=v_k\cos\theta$", size=EQUATION_FONT_SIZE)
    label(ax, 1.75, 1.04, r"$v_{\phi,0}=v_c(R_0)+v_k\sin\theta\sin\psi$", size=EQUATION_FONT_SIZE)
    label(ax, 1.75, 0.55, r"$j_{z,0}=R_0v_{\phi,0}$", size=EQUATION_FONT_SIZE)

    label(ax, 7.65, 3.32, "First downward crossing", size=11, color=ORANGE)
    label(ax, 7.65, 2.83, r"$z(t_{\rm land})=0$", size=EQUATION_FONT_SIZE)
    label(ax, 7.65, 2.28, r"$v_z(t_{\rm land})<0$", size=EQUATION_FONT_SIZE)
    label(ax, 7.65, 1.72, r"$\tau=t_{\rm land}-t_0$", size=EQUATION_FONT_SIZE)

    # Below the disk: deposited sources and their angular-momentum-driven flow.
    # Here j_land is the source-weighted value, using the existing disk notation.
    label(ax, -8.9, -1.55, "Disk response", size=11.5, color=BLUE)
    label(ax, -8.9, -2.12,
          r"$\Sigma_g v_R\,\dfrac{d j_{\rm disk}}{dR}"
          r"=\dot\Sigma_{\rm land}\left(j_{\rm land}-j_{\rm disk}\right)$", size=EQUATION_FONT_SIZE)
    label(ax, -8.9, -2.76,
          r"$j_{\rm disk}=Rv_c,\quad j_{\rm disk}'>0,\qquad"
          r"j_{\rm land}=\dot J_{\rm land}/\dot\Sigma_{\rm land}$", size=EQUATION_FONT_SIZE)

    label(ax, 1.0, -1.65, "Delayed deposition", size=11.5, color=BLUE)
    label(ax, 1.0, -2.22,
          r"$\dot\Sigma_{\rm land}(R,t)=\int_0^\infty K_{\rm tot}(R,\tau)"
          r"\,\dot M_w(t-\tau)\,d\tau$", size=EQUATION_FONT_SIZE)
    label(ax, 1.0, -3.03,
          r"$K_{{\rm tot},ij}=\dfrac{\sum_{a\in ij}w_a(m_{{\rm land},a}/m_{0,a})}"
          r"{(\sum_{\rm all}w_a)\,\Delta A_i\,\Delta\tau_j},\qquad"
          r"\dot M_w=\eta_{\rm nuc}\dot M_{\rm nuc}$", size=EQUATION_FONT_SIZE)


def make_figure(outfile="fountain_cycle_cartoon", show=False):
    """Save one PDF; optionally display it when explicitly requested in Python."""
    style = {
        "text.usetex": False,
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 12,
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
    }
    with mpl.rc_context(style):
        fig = plt.figure(figsize=(16.8, 9.5), facecolor="white")
        ax = fig.add_axes([0.025, 0.035, 0.95, 0.93])
        ax.set(xlim=(-9.5, 11.1), ylim=(-3.55, 8.5), aspect="equal")
        ax.axis("off")
        draw_scene(ax)
        annotate_physics(ax)

        output = Path(outfile)
        if output.suffix.lower() != ".pdf":
            output = Path(f"{output}.pdf")
        fig.savefig(output, format="pdf", facecolor="white",
                    metadata={"Subject": "Nuclear fountain: launch, orbit, CGM exchange, and deposition"})
        if show:
            plt.show()
        else:
            plt.close(fig)
    return fig


if __name__ == "__main__":
    make_figure()
