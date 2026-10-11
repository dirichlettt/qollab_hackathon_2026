# Quantum Brachistochrone on a Qubit — Qollab Hackathon 2026

Submission for the 2026 Qollab Hackathon. Built with [Qiskit](https://qiskit.org) and
released under the MIT license.

## What it does

This project reproduces, numerically verifies, and turns into runnable Qiskit circuits the
single-qubit *restricted* quantum brachistochrone of

> Jing Yang & Adolfo del Campo, *Minimum-Time Quantum Control and the Quantum Brachistochrone
> Equation*, [arXiv:2204.12792](https://arxiv.org/abs/2204.12792), Section VIII, Example A.

The question the paper asks: **what is the fastest way to steer a qubit from one state to
another when you are not allowed to use every control?** The quantum speed limit tells you the
minimum time when any Hamiltonian is allowed. Real hardware is restricted, and the restricted
minimum time is longer. The quantum brachistochrone (QB) equation gives the time-optimal
Hamiltonian under those restrictions. Yang & del Campo show that the classic CHKO form of the
QB equation is incomplete: it misses an endpoint condition (their Eq. 55) that any true
time-extremal trajectory must satisfy.

Example A is the simplest non-trivial case: one qubit, bounded Hamiltonian norm `omega`, and
the control `sigma_z` is forbidden. The closed-subalgebra solution is

```
H(t) = omega * exp(i l1 Z t) Y exp(-i l1 Z t)
U(t) = exp(i l1 Z t) * exp(-i (omega Y + l1 Z) t)
```

where `l1` is a Lagrange multiplier fixed by the forbidden control.

## What the code does

`qb_example_A.py` runs five steps and prints every check:

1. **Builds U(T) as a Qiskit circuit** (`H`, a `PauliEvolutionGate` for
   `exp(-i (omega Y + l1 Z) T)`, then `RZ`) and checks it against the analytic unitary.
   The evolution gate is synthesised with `MatrixExponential` so it stays exact after
   transpiling to the `rz, sx, x, cx` basis (the default Trotter split is not exact because
   `Y` and `Z` do not commute).
2. **Verifies the dynamics**: `H = i dU/dt U^dagger`, the norm and trace constraints, the
   CHKO equation `F' + i[H, F] = 0`, and the initial condition on `F(0)`.
3. **Verifies the missing endpoint condition** (Eq. 55 / Eq. 86) against the paper's closed
   form `l1 * cos(2 * Lambda1 * T)`.
4. **Reproduces Fig. 2** of the paper (`qb_fig2.png`): overlap with the initial state and
   the endpoint condition as a function of `omega T / pi` and the rescaled multiplier. Only
   points on the black contour are locally time-extremal.
5. **Solves the prescribed-target problem** (Eqs. 97-105): for a target
   `|psi_f(phi, Omega_B)>` it finds the allowed integer branches `(k, l)`, computes
   `T(k, l)` from Eq. 103, and confirms the result by simulating the Qiskit circuit.

`bell.py` is a two-line sanity check that Qiskit and Aer are installed (Bell state, 1000 shots).

### Interactive app

`app.py` is a Streamlit front end over the same functions. Pick ω and a target state in the
sidebar and the app shows the restricted versus unrestricted minimum time, every candidate
branch `(k, l)` with its fidelity, the Qiskit circuit, an Aer measurement run, the Bloch-vector
trajectory, and a button that regenerates Fig. 2.

```bash
streamlit run app.py
```

### Sample output

```
circuit vs analytic fidelity           : 1.000000000000
transpiled (rz,sx,x,cx) fidelity       : 1.000000000000   (5 gates)
|F' + i[H,F]|                          : 5.10e-09
...
global minimum: T = 0.064342  = 1.165 x free time (Thm 1: >= 1)
Qiskit circuit fidelity with |psi_f>: 1.0000000000

   ┌───┐┌──────────────┐┌─────────┐┌─────────────┐┌──────────────┐
q: ┤ H ├┤ Rx(-0.61073) ├┤ Ry(π/2) ├┤ Rx(0.61073) ├┤ Rz(-0.90079) ├
   └───┘└──────────────┘└─────────┘└─────────────┘└──────────────┘
```

The headline number: with `sigma_z` forbidden, reaching this target takes **1.165 times** the
unrestricted speed-limit time, in agreement with Theorem 1 of the paper (restricted time is
never shorter than free time).

## How to run

Python 3.10+ is required.

```bash
git clone https://github.com/dirichlettt/qollab_hackathon_2026.git
cd qollab_hackathon_2026
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python bell.py           # sanity check: Qiskit + Aer work
python qb_example_A.py   # full reproduction; writes qb_fig2.png
```

Everything runs on the local Qiskit Aer / statevector simulator. No account, API key, or
hardware access is needed. The same circuits run unchanged on Qollab's free simulators,
including IonQ's simulator, but results from those neither add to nor change the output.

Runtime is about a minute, almost all of it spent on the Fig. 2 grid. Lower `n_l` and `n_T`
in `make_figure()` for a faster run.

## Accessibility

The Streamlit app follows standard web accessibility practice:

- Every control has a visible label and a help tooltip; no control is identified by colour
  or position alone.
- All controls are native widgets, so they are keyboard navigable and announced by screen
  readers, and Streamlit respects the operating system's light or dark preference.
- Numbers are shown as text in metrics and tables, never only inside a plot.
- The Bloch trajectory plot distinguishes the three components by line style as well as
  colour, and each figure has a text caption stating what it shows and the key values.
- The Fig. 2 reproduction marks the extremal set with a solid black contour line, not only
  with colour, and uses perceptually uniform colormaps.

The command-line script prints every result as plain text so it can be read aloud or copied.

## How to extend it

- **Example B** (Section VIII.2): two qubits with all `sigma_x` terms forbidden. The paper
  gives the optimal `H = (omega / sqrt 2) sigma_y x sigma_y` and `T = sqrt 2 Omega_B / omega`.
  `qb_circuit()` and `endpoint_quantities()` generalise directly; only the generator and the
  forbidden-operator check change.
- **Other forbidden operators**: replace `Z` in the generator and in the `Tr(H X1)` check.
- **Noisy simulation**: `qb_circuit()` returns an ordinary `QuantumCircuit`, so it can be
  transpiled and run on an Aer noise model or on Qollab's IonQ simulator with measurements
  added.
- **Sweeps**: `solve_known_target()` is pure Python over `(phi, Omega_B)`, so it can be
  mapped over a grid of targets to chart the restricted speed limit.

## Originality disclosure

All code in this repository was written during the hackathon window. No templates or forks
were used. The physics, equations, and Fig. 2 layout follow the paper cited above, and
equation numbers in the code comments refer to that paper.

## License

MIT. See [LICENSE](LICENSE).
