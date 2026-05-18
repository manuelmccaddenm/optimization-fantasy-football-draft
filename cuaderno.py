# %% [markdown]
# # El draft de fantasy football como problema de optimizacion
#
# Los algoritmos viven en `optimizacion.py`; `construir_datos.py` genera
# `data/jugadores.csv`, `data/mu.csv` y `data/sigma.csv`. Este cuaderno solo
# carga esos datos y corre los experimentos.

# %%
import os
import numpy as np
import pandas as pd

from optimizacion import (
    LeagueConfig, HyperParams, Preferences,
    simulate_draft, roster_summary, draw_board, plot_pick_decision,
)
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)

players = pd.read_csv("data/jugadores.csv")
mu = np.loadtxt("data/mu.csv", delimiter=",")
Sigma = np.loadtxt("data/sigma.csv", delimiter=",")

LEAGUE = LeagueConfig()
HP = HyperParams()
PREFS = Preferences()
OWN_OBJECTIVE = {"markowitz": "U1", "upside": "U2", "adp": "pts", "vbd": "pts"}
SLOT = 4

# %% [markdown]
# ## Un draft (puesto 5 de 12)

# %%
for strategy in ["markowitz", "upside", "adp", "vbd"]:
    rosters = simulate_draft(players, mu, Sigma, LEAGUE, HP, PREFS,
                             your_slot=SLOT, your_strategy=strategy,
                             rng=np.random.default_rng(0))
    yours = rosters[SLOT]
    pts, risk, U1, U2 = roster_summary(players, mu, Sigma, HP, yours)
    own = {"U1": U1, "U2": U2, "pts": pts}[OWN_OBJECTIVE[strategy]]
    print(f"\n=== {strategy.upper()} ===  "
          f"pts {pts:.1f}  riesgo {risk:.1f}  U1 {U1:.1f}  U2 {U2:.1f}  "
          f"propio {own:.1f}")
    print([players.loc[i, "name"] for i in yours])

# %% [markdown]
# ## Tablero serpiente

# %%
boards = {s: simulate_draft(players, mu, Sigma, LEAGUE, HP, PREFS,
                            your_slot=SLOT, your_strategy=s,
                            rng=np.random.default_rng(0))
          for s in ("markowitz", "upside")}
fig, axs = plt.subplots(1, 2, figsize=(LEAGUE.n_teams * 2.4,
                                       LEAGUE.roster_size * 0.55))
for ax, s in zip(axs, ("markowitz", "upside")):
    draw_board(boards[s], players, SLOT, LEAGUE.roster_size, ax=ax,
               title_suffix=f"  [{s.upper()}]")
fig.tight_layout()
fig.savefig("figs/snake_board.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Monte Carlo (puesto aleatorio)

# %%
results = {s: [] for s in ["markowitz", "upside", "adp", "vbd"]}
for k in range(6):
    slot = int(np.random.default_rng(1000 + k).integers(0, LEAGUE.n_teams))
    for strategy in results:
        rosters = simulate_draft(players, mu, Sigma, LEAGUE, HP, PREFS,
                                 your_slot=slot, your_strategy=strategy,
                                 rng=np.random.default_rng(1000 + k))
        pts, risk, U1, U2 = roster_summary(players, mu, Sigma, HP,
                                           rosters[slot])
        results[strategy].append((slot, pts, risk, U1, U2))

resumen = {}
for s, rows in results.items():
    pts = np.array([r[1] for r in rows])
    rk = np.array([r[2] for r in rows])
    U1 = np.array([r[3] for r in rows])
    U2 = np.array([r[4] for r in rows])
    own = {"U1": U1, "U2": U2, "pts": pts}[OWN_OBJECTIVE[s]]
    resumen[s] = (own.mean(), own.std(), pts.mean(), rk.mean())
    print(f"{s:10s} propio {own.mean():8.1f} +- {own.std():5.1f}   "
          f"pts {pts.mean():7.1f}  riesgo {rk.mean():7.1f}")

# %%
etq = {"markowitz": "Markowitz", "upside": "Upside",
       "vbd": "VBD", "adp": "ADP"}
orden = ["markowitz", "upside", "vbd", "adp"]
col = ["#2c7fb8", "#d95f02", "#7570b3", "#999999"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
a1.bar([etq[s] for s in orden], [resumen[s][0] for s in orden],
       yerr=[resumen[s][1] for s in orden], color=col, capsize=5,
       edgecolor="black", linewidth=0.6)
a1.set_ylabel("Valor del criterio propio")
a1.set_title("Utilidad media bajo la propia metrica")
for i, s in enumerate(orden):
    a1.text(i, resumen[s][0] + resumen[s][1] + 40,
            f"{resumen[s][0]:.0f}", ha="center", fontsize=9)
for i, s in enumerate(orden):
    a2.scatter(resumen[s][3], resumen[s][2], s=120, color=col[i],
               edgecolor="black", linewidth=0.6, zorder=3)
    a2.annotate(etq[s], (resumen[s][3], resumen[s][2]),
                textcoords="offset points", xytext=(8, 6), fontsize=10)
a2.set_xlabel(r"riesgo del equipo $\lambda R^{\top}\Sigma R$")
a2.set_ylabel("puntos esperados")
a2.set_title("Puntos esperados frente a riesgo")
a2.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("figs/mc_comparison.png", dpi=130, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## La decision en el plano (riesgo, rendimiento)

# %%
plot_pick_decision(players, mu, Sigma, LEAGUE, HP, PREFS,
                   your_slot=0, k=4, n_cand=5,
                   save_path="figs/pick_decision.png")
