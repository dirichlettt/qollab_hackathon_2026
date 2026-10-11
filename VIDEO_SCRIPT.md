# Video script: 2-minute final submission

Quantum Brachistochrone Explorer, Qollab Hackathon 2026.
About 290 spoken words, roughly 2 minutes at a calm pace.
Each block gives the on-screen visual, then the narration.

## [0:00-0:15] Hook

**Visual:** title card, then the Bloch sphere with a short arc between two states.

> How fast can you move a qubit from one state to another? If you can use any
> Hamiltonian, the quantum speed limit gives the answer. But real hardware can't do
> everything. Some controls are simply not available. So what is the fastest path when
> one control is forbidden?

## [0:15-0:35] The physics

**Visual:** paper title page for arXiv:2204.12792, then zoom to the quantum
brachistochrone equation and the endpoint condition (Eq. 55).

> That is the restricted quantum brachistochrone problem. Our project builds on a 2022
> paper by Yang and del Campo. They showed that the textbook brachistochrone equation is
> incomplete. It misses an endpoint condition that every true time-optimal trajectory
> has to satisfy. We took their single-qubit example, where the σ_z control is
> forbidden, and made it runnable.

## [0:35-1:05] What we built

**Visual:** screen recording of the Streamlit app. Pick ω in the sidebar, choose the
"Any initial and final state" mode, drag the final-state sliders.

> This is our Quantum Brachistochrone Explorer, built on Qiskit. You pick the energy
> bound and any initial and final state on the Bloch sphere. The app solves the
> brachistochrone equation, finds every extremal branch, and reports the restricted
> minimum time next to the unrestricted speed-limit time. Then it does something a
> formula alone can't do: it compiles the optimal evolution into a real Qiskit circuit,
> runs it on the Aer simulator, and checks the fidelity against the target.

## [1:05-1:30] Proof it works

**Visual:** terminal output of `python qb_example_A.py` scrolling, pause on the
fidelity line, then cut to the Fig. 2 reproduction (`qb_fig2.png`) with the black
contour.

> Every step is verified numerically. The circuit matches the analytic unitary to
> machine precision, even after transpiling to native gates. We check the
> brachistochrone equation, the norm and trace constraints, and the missing endpoint
> condition against the paper's closed form. And we reproduce Figure 2 of the paper:
> only points on the black contour are time-extremal.

## [1:30-1:50] The headline result

**Visual:** the "Target and minimum times" metrics panel in the app, with the ratio
highlighted.

> For the paper's target state, the restricted path takes 1.165 times longer than the
> free speed limit. That is exactly what their Theorem 1 predicts: restricting your
> controls can never make you faster. Our Qiskit circuit reaches the target with
> fidelity 1.

## [1:50-2:00] Close

**Visual:** GitHub repo page, then end card with the project name.

> Everything runs locally with no hardware access needed, and the same circuits run
> unchanged on Qollab's simulators. The code is open source under MIT. Thanks for
> watching.

## Recording notes

- The 1.165 ratio and the five-gate circuit come from the README sample output
  (target φ = 2.73154, Ω_B = 0.55230, ω = 10). Show that exact run, not a random
  slider setting.
- The Fig. 2 regeneration takes about a minute. Pre-render it and cut to the saved
  `qb_fig2.png` instead of pressing the button live.
- Record the app at a wide window so the sidebar and the metrics panel fit in one frame.
