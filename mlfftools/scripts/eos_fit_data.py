#! /usr/bin/env python3
import json
from pathlib import Path

import pandas as pd
from scipy.optimize import curve_fit
from ase.units import GPa
import numpy as np
from scipy import interpolate as si
from matplotlib import pyplot as plt
from rich import print as echo
import typer


def vinet(V, E0, B0, BP, V0):
    "Vinet equation from PRB 70, 224107"

    X = (V / V0) ** (1 / 3)
    eta = 3 / 2 * (BP - 1)

    E = E0 + 2 * B0 * V0 / (BP - 1) ** 2 * (
        2 - (5 + 3 * BP * (X - 1) - 3 * X) * np.exp(-eta * (X - 1))
    )
    return E


def vinet_pressure(V, B0, BP, V0):
    """Eq. (4.1) in P. Vinet et al., Phys Rev B 35, 1945 (1987)."""
    X = (V / V0) ** (1 / 3)
    eta = 3 / 2 * (BP - 1)
    P = 3 * B0 * (1 - X) / X ** 2 * np.exp(eta * (1 - X))
    return P


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file: Path,
    plot: bool = False,
    kind: str = "quadratic",
):
    """fit Vinet EOS to volume/energy data and write fit parameters to file"""
    df = pd.read_csv(file, comment="#")
    x = df.volume / df.N
    y = df.energy / df.N
    y0 = y.min()
    p = df.pressure  # / GPa

    bounds = ([-1, 0, -10, 0], [1, 5, 10, 100])
    popt_e, _ = curve_fit(vinet, x, y - y0, bounds=bounds)

    bounds = ([0, -10, 0], [5, 10, 100])
    popt_p, _ = curve_fit(vinet_pressure, x, p, bounds=bounds)

    E0, B0, BP, V0 = popt_e
    E0 += y0
    kwargs_e = {"E0": E0, "B0": B0, "BP": BP, "V0": V0}

    echo("Results from fitting *energies* to Vinet equation of states:")
    echo(json.dumps(kwargs_e, indent=1))

    B0, BP, V0 = popt_p
    kwargs_p = {"E0": E0, "B0": B0, "BP": BP, "V0": V0}

    echo("Results from fitting *pressures* to Vinet equation of states:")
    echo(json.dumps(kwargs_p, indent=1))

    outfile = file.stem + "_e.json"
    echo(f"... write energy fit to {outfile}")
    json.dump(kwargs_e, open(outfile, "w"), indent=1)

    outfile = file.stem + "_p.json"
    echo(f"... write pressure fit to {outfile}")
    json.dump(kwargs_p, open(outfile, "w"), indent=1)

    if plot:
        fig, ax = plt.subplots()
        # plot interpolated pressure
        f_p = si.interp1d(x, p, kind=kind)
        # ax = df.plot(x="volume", y="pressure", style="*")
        _y = 1e3 * (y - y0)
        ax.plot(x, _y, marker="*", lw=0, color="k")
        # df.plot(x="volume", y="energy", style="*", ax=ax, color="k")
        tax = ax.twinx()
        tax.plot(x, p / GPa, marker=".", lw=0, color="C3")
        # df.plot(x="volume", y="pressure", style=".", ax=tax, color="C3")
        _x = np.linspace(x.min(), x.max(), num=1000)
        tax.plot(_x, f_p(_x) / GPa, label="interpolation", zorder=0, color="C3")

        # plot vinet pressure
        _y = vinet_pressure(_x, *popt_e[1:]) / GPa
        tax.plot(_x, _y, color="C0")

        tax.axhline(0, c="k", lw=1)

        ax.plot(_x, 1e3 * vinet(_x, *popt_e), color="k")
        ax.plot(_x, 1e3 * vinet(_x, 0, *popt_p), lw=1, color="teal")

        ax.legend(["Calculation", "EOS fit to energy", "EOS fit to pressure"])
        tax.legend(["Pressure", "Fit", "Pressure from energy"])

        ax.set_xlabel(r"Volume per atom (${\rm \AA}^3$)")
        ax.set_ylabel("Energy per atom (meV)")
        tax.set_ylabel("Pressure (GPa)")

        outfile = file.stem + ".pdf"
        echo(f"... save plot to {outfile}")
        fig.savefig(outfile)


if __name__ == "__main__":
    app()
