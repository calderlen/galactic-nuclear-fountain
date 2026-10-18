#!/usr/bin/env python3
"""Larger, two-sided alternative to fountain_cycle_cartoon.pdf.

This is an equation-annotated illustration, not an integrated orbit. The lower
fountain is the exact z-reflection of the upper one. The production integrator
follows the upper hemisphere; no integrator or kernel normalization is changed.
All math retains the original figure's notation, with a common 21-point size.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from fountain_cycle_cartoon import BLUE, HALO, INK, MUTED, ORANGE, PALE_BLUE, arrow, label, leader, orbit, trajectory


MAIN = np.array(((1.1, 0.18), (1.5, 14.7), (9.6, 0.18)))
STAGES = (0.25, 0.5, 0.77)
TEXT_SCALE = 1.4
EQUATION_FONT_SIZE = 21


def text(ax, x, y, value, *, color=INK, ha="left", size=12):
    """One font size for every mathematical annotation."""
    return label(ax, x, y, value, color=color, ha=ha, size=EQUATION_FONT_SIZE if "$" in value else size * TEXT_SCALE)


def heading(ax, x, y, value, *, color=BLUE):
    return text(ax, x, y, value, color=color, size=12.5)


def parcel(ax, point):
    # Identical marker footprints: sizes do not encode increasing cross-section.
    ax.add_patch(Ellipse(point, 0.62, 0.62, facecolor=PALE_BLUE, edgecolor=BLUE, lw=0.8, zorder=12))
    ax.add_patch(Ellipse(point, 0.41, 0.41, facecolor=ORANGE, edgecolor="white", lw=1, zorder=13))


def capture_streams(ax, point, sign):
    # Illustrative capture streamlines in the parcel frame, not CGM velocities
    # in the galactic frame and not a resolved hydrodynamic mixing layer.
    offsets = np.array(((0.85, 0.54), (1.23, 0.55), (1.67, 0.51), (1.01, 0.19), (1.42, 0.22), (1.84, 0.19)))
    offsets[:, 1] *= sign
    gas = point + offsets
    ax.scatter(*gas.T, s=[15, 21, 11, 13, 17, 10], c=BLUE, alpha=0.6, edgecolors="none", zorder=10)
    for start, end, bend in [((1.12, 0.48), (0.23, 0.23), 0.15), ((1.44, 0.18), (0.32, 0.01), -0.07)]:
        arrow(ax, point + np.array(start) * (1, sign),
              point + np.array(end) * (1, sign), BLUE,
              lw=1.1, size=9, rad=sign * bend, zorder=14)


def draw_fountains(ax):
    ax.add_patch(Ellipse((0, 0), 23, 20.2, facecolor=HALO, edgecolor="none", zorder=0))
    examples = [
        (10, 0, 9.1, -4.2, "#CF956C"),
        (26, 35, 6.6, -5.0, "#D9A782"),
        (43, 0, 4.0, -5.7, "#E1B795"),
        (60, -20, 2.1, -6.4, "#E8C6A9"),
    ]
    for sign in (1, -1):
        for theta, psi, height, landing_x, color in examples:
            slope = np.tan(np.deg2rad(theta)) * np.cos(np.deg2rad(psi))
            start = np.array((-1.1, sign * 0.18))
            control = start + (-height * slope, sign * height)
            orbit(ax, (start, control, (landing_x, start[1])), color, lw=1.25, arrows=(0.085, 0.25, 0.78), zorder=2)
        path = MAIN * (1, sign)
        orbit(ax, path, ORANGE, lw=2.6, arrows=(0.13, 0.37, 0.65, 0.9), zorder=5)
        for t in STAGES:
            parcel(ax, trajectory(t, path))
        # Use the two hemispheres for complementary detail: capture above,
        # force vectors below. Only the fountain geometry is duplicated.
        if sign > 0:
            capture_streams(ax, trajectory(0.5, path), sign)

    # Preserve a shared disk and launch ring, rather than making two panels.
    ax.add_patch(Ellipse((0, 0), 22, 0.52, facecolor="#E3E4DF", edgecolor="#929B98", lw=1, zorder=15))
    ax.add_patch(Ellipse((0, 0), 2.85, 0.78, facecolor="#CDD2CB", edgecolor="none", zorder=16))
    ax.add_patch(Ellipse((0, 0), 2.2, 0.39, facecolor="#F9E0CD", edgecolor=ORANGE, lw=1.8, zorder=17))
    ax.add_patch(Ellipse((0, 0), 1.72, 0.22, facecolor="#F7F6F0", edgecolor="none", zorder=18))
    for sign in (1, -1):
        ax.scatter([1.1, 9.6], [sign * 0.18, sign * 0.18], s=[35, 65], c=ORANGE, edgecolors="white", lw=0.8, zorder=19)

    arrow(ax, (8.0, 0), (4.7, 0), BLUE, lw=1.3, size=10, zorder=19)
    text(ax, 6.35, 0.53, r"$j_{\rm land}<j_{\rm disk}\ \Rightarrow\ v_R<0$", color=BLUE, ha="center")
    leader(ax, "Galactic disk", (-9.1, 0.1), (-9.1, 0.9), ha="center", size=12 * TEXT_SCALE)
    text(ax, -0.15, -0.90, "Circumnuclear\nring", ha="center", size=11.5)
    text(ax, 1.55, -0.55, r"$m_0,\ t_0$")
    arrow(ax, (0, 1.55), (0, 3.05), MUTED, lw=0.8, size=9)
    text(ax, -0.12, 3.30, r"$z$")
    ax.plot([-1.1, 0], [0.69, 0.69], color=MUTED, lw=0.75, zorder=19)
    ax.plot([0, 0], [0.62, 0.76], color=MUTED, lw=0.75, zorder=19)
    ax.plot([-1.1, -1.1], [0.62, 0.76], color=MUTED, lw=0.75, zorder=19)
    text(ax, -0.55, 1.06, r"$R_{\rm nuc}$", ha="center")


def upper_annotations(ax):
    heading(ax, -11.2, 10.25, "Orbit in the galactic potential", color=MUTED)
    text(ax, -11.2, 9.58, r"$\Phi=\Phi_{\rm MN}+\Phi_{\rm H}+\Phi_{\rm NFW}$")
    text(ax, -11.2, 8.86, r"$\dot R=v_R,\quad \dot z=v_z,\quad v_c^2=R\partial_R\Phi(R,0)$")
    text(ax, -11.2, 8.07, r"$\dot v_R=-\partial_R\Phi+\dfrac{j_z^2}{R^3}-\Gamma v_R$")
    text(ax, -11.2, 7.29, r"$\dot v_z=-\partial_z\Phi-\Gamma v_z$")

    heading(ax, 3.5, 10.75, "Rotating CGM")
    text(ax, 3.5, 10.10,
         r"$\mathbf{v}_{\rm CGM}=\beta v_c(R)\hat{\mathbf{e}}_\phi,"
         r"\qquad \rho_{\rm CGM}=\mathrm{const.}$", color=BLUE)
    arrow(ax, (0.1, 9.35), (2.6, 9.4), BLUE, lw=1.1, rad=-0.1)
    arrow(ax, (9.5, 9.45), (12.0, 9.28), BLUE, lw=1.1, rad=-0.1)

    heading(ax, -11.2, 5.96, "Launch distribution", color=ORANGE)
    text(ax, -11.2, 5.30, r"$p(R_0)\propto R_0 e^{-(R_0-R_{\rm nuc})^2/(2\sigma_R^2)}$")
    text(ax, -11.2, 4.57, r"$p(v_k)\propto v_k^2e^{-v_k^2/(2h_v^2)}$")
    text(ax, -11.2, 3.84, r"$p(\theta)\propto\sin\theta\,e^{-\theta^2/(2\sigma_\theta^2)}$")
    text(ax, -11.2, 3.11, r"$0\leq\theta<\pi/2,\qquad \psi\sim\mathcal{U}(0,2\pi)$")
    text(ax, -11.2, 2.52, "Angle measured from each outward vertical", color=MUTED, size=10.5)
    text(ax, -6.7, 5.12, "Projected launch directions", color=MUTED)
    text(ax, -6.7, 4.58, r"$v_{R,0}/|v_{z,0}|=\tan\theta\cos\psi$", color=MUTED)

    heading(ax, 2.65, 3.83, "1  Launch into both hemispheres", color=ORANGE)
    text(ax, 2.65, 3.20, r"$v_{R,0}=v_k\sin\theta\cos\psi$")
    text(ax, 2.65, 2.59, r"$v_{z,0}=\pm v_k\cos\theta$")
    text(ax, 2.65, 1.98, r"$v_{\phi,0}=v_c(R_0)+v_k\sin\theta\sin\psi$")

    heading(ax, -4.0, 7.11, "2  Relative motion and drag")
    text(ax, -4.0, 6.49, r"$u=|\mathbf{v}-\mathbf{v}_{\rm CGM}|$")
    text(ax, -4.0, 5.77, r"$\Gamma=\dfrac{\dot m}{m}+\dfrac{C_D}{2}\dfrac{\rho_{\rm CGM}\sigma u}{m}$")

    heading(ax, 5.9, 8.83, "3  Cumulative mass growth")
    text(ax, 5.9, 8.20, r"$\dot m=\epsilon\rho_{\rm CGM}\sigma u$")
    text(ax, 5.9, 7.48, r"$m(t)=m_0+\int_{t_0}^{t}\dot m(t')\,dt'$")
    text(ax, 5.9, 6.78, r"$\sigma=\mathrm{const.}$", color=BLUE)
    text(ax, 5.9, 6.40, "Capture and drag act throughout flight", color=MUTED, size=11)
    apex = trajectory(0.5, MAIN)
    arrow(ax, apex + (-0.6, 0.85), apex + (0.75, 0.85), ORANGE, lw=1.3, size=10)
    text(ax, apex[0] + 0.1, apex[1] + 1.20, r"$v_R$", color=ORANGE, ha="center")
    leader(ax, r"$m(t),\ j_z(t)$", trajectory(0.77, MAIN) + (0.21, 0.21), (6.7, 6.03), size=EQUATION_FONT_SIZE)

    # A single crossing condition covers both hemispheres: the returning
    # vertical velocity has the opposite sign to the launch velocity.
    heading(ax, 8.70, 4.90, "4  First return to the disk", color=ORANGE)
    text(ax, 8.70, 4.20, r"$z(t_{\rm land})=0$")
    text(ax, 8.70, 3.49, r"$v_{z,0}\,v_z(t_{\rm land})<0$")
    text(ax, 8.70, 2.78, r"$R(t_{\rm land})=R_{\rm land}$")
    text(ax, 8.70, 2.07, r"$\tau=t_{\rm land}-t_0$")
    text(ax, 10.05, 0.77, r"$m_{\rm land},\quad j_{\rm land}$")


def lower_annotations(ax):
    heading(ax, 2.65, -1.8, "Angular-momentum exchange")
    text(ax, 2.65, -2.44, r"$\dot j_z=\Gamma(\beta Rv_c-j_z)$")
    text(ax, 2.65, -3.05, r"$j_z=Rv_\phi$")
    text(ax, 2.65, -3.69, r"$j_z>\beta Rv_c\ \Rightarrow\ \dot j_z<0$")
    text(ax, 2.65, -4.31, r"$j_z<\beta Rv_c\ \Rightarrow\ \dot j_z>0$")

    lower = MAIN * (1, -1)
    outward = trajectory(0.25, lower)
    arrow(ax, outward + (-0.48, 0.15), outward + (-0.48, 1.0), BLUE, lw=1.25, size=9, zorder=14)
    text(ax, outward[0] - 0.85, outward[1] + 1.52, r"$-\Gamma v_z>0$", color=BLUE, ha="right")
    text(ax, -1.9, -5.77, r"$v_z<0$", color=ORANGE)
    text(ax, -1.9, -6.34, "Outward motion below the disk", color=MUTED, size=11)

    apex = trajectory(0.5, lower)
    arrow(ax, apex + (-0.38, 0), apex + (-1.7, 0), BLUE, lw=1.3, size=10, zorder=14)
    text(ax, apex[0] - 1.68, apex[1] - 0.4, r"$-\Gamma v_R$", color=BLUE)
    arrow(ax, apex + (-0.13, 0.38), apex + (-0.6, 1.25), MUTED, lw=1.1, size=10, zorder=14)
    text(ax, apex[0] + 0.05, apex[1] + 0.95, r"$-\nabla\Phi$", color=MUTED)
    heading(ax, 1.85, -8.40, "Vertical turnaround", color=MUTED)
    text(ax, 1.85, -9.08, r"$v_z=0,\qquad z\,\dot v_z<0$")

    heading(ax, -11.2, -1.70, "5  Deposition and disk response")
    text(ax, -11.2, -2.42,
         r"$\Sigma_gv_R\dfrac{d j_{\rm disk}}{dR}"
         r"=\dot\Sigma_{\rm land}(j_{\rm land}-j_{\rm disk})$")
    text(ax, -11.2, -3.22, r"$j_{\rm disk}=Rv_c,\quad j'_{\rm disk}>0$")
    text(ax, -11.2, -3.89, r"$j_{\rm land}=\dot J_{\rm land}/\dot\Sigma_{\rm land}$")

    heading(ax, -11.2, -4.85, "Mass-weighted return kernel")
    text(ax, -11.2, -6.00,
         r"$K_{{\rm tot},ij}=\dfrac{\sum_{a\in ij}w_a(m_{{\rm land},a}/m_{0,a})}"
         r"{(\sum_{\rm all}w_a)\,\Delta A_i\,\Delta\tau_j}$")
    text(ax, -11.2, -7.02, r"$\dot M_w=\eta_{\rm nuc}\dot M_{\rm nuc}$")
    heading(ax, -11.2, -8.32, "Delayed deposition")
    text(ax, -11.2, -9.07, r"$\dot\Sigma_{\rm land}(R,t)=\int_0^\infty K_{\rm tot}(R,\tau)" r"\dot M_w(t-\tau)\,d\tau$")

    heading(ax, 8.3, -3.90, "Momentum transfer")
    text(ax, 8.3, -4.70,
         r"$\left.\dot{\mathbf{v}}\right|_{\rm acc}"
         r"=\dfrac{\dot m}{m}(\mathbf{v}_{\rm CGM}-\mathbf{v})$")
    text(ax, 8.3, -5.65,
         r"$\left.\dot{\mathbf{v}}\right|_{\rm drag}"
         r"=-\dfrac{C_D}{2}\dfrac{\rho_{\rm CGM}\sigma u}{m}"
         r"(\mathbf{v}-\mathbf{v}_{\rm CGM})$")

    heading(ax, -11.2, -10.91, "Midplane symmetry", color=MUTED)
    text(ax, -11.2, -11.58, r"$\Phi(R,z)=\Phi(R,-z),\qquad (z,v_z)\leftrightarrow(-z,-v_z)$", color=MUTED)
    text(ax, 1.8, -11.58, r"$R,\ v_R,\ j_z,\ m\ \mathrm{unchanged}$", color=MUTED)


def make_figure(outfile="fountain_cycle_cartoon_bipolar.pdf"):
    """Write only the alternative PDF; do not regenerate the original figure."""
    style = {
        "text.usetex": False, "font.family": "STIXGeneral",
        "mathtext.fontset": "stix", "font.size": 12,
        "pdf.fonttype": 42, "axes.unicode_minus": False,
    }
    with mpl.rc_context(style):
        fig = plt.figure(figsize=(24, 21.5), facecolor="white")
        ax = fig.add_axes([0.025, 0.025, 0.95, 0.95])
        ax.set(xlim=(-12, 15), ylim=(-12.3, 11.4), aspect="equal")
        ax.axis("off")
        draw_fountains(ax)
        upper_annotations(ax)
        lower_annotations(ax)
        output = Path(outfile)
        if output.suffix.lower() != ".pdf":
            output = Path(f"{output}.pdf")
        fig.savefig(output, format="pdf", facecolor="white", metadata={
            "Subject": "Bipolar nuclear fountain: symmetric launch, CGM interactions, return, and disk response",
        })
        plt.close(fig)
    return fig


if __name__ == "__main__":
    make_figure()
