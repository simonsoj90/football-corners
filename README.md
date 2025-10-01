# football-corners
Model for betting on total corners in football matches

- Trains a hierarchical model for **total corners** (and optionally a joint model with **goals**).
- Exports lightweight **artifacts** (posterior draws + metadata).
- Loads live odds to price **Over/Under (corners & goals)** and compute **EV** and **Kelly stakes**.
- Optionally prices a **parlay (corners × goals)** with a Gaussian copula correlation.
- Caps total stakes so you **never exceed your bankroll** for the slate.

## Outline
1) **Train the model**  
   Open `v3_corners_nb.ipynb`, run end-to-end. At the end, run the **artifact export** cell (already in the notebook) so you get:
   ```
   corners_model_draws.npz, _teams.json, _divs.json, _zstats.json
   ```

2) **Prepare match inputs**  
   Create/append `data/matchday_inputs.csv`:
   - Odds can be **fractional** (`11/10`, `1/1`) or **decimal** (`2.1`).
   - `parlay_price` optional (if blank we use product of leg prices).
   - `odds_*` columns are for feature signals (1×2 and O/U2.5) and can be left blank—model falls back to z-stats.

3) **Price & stake**  
   Open `daily_pricer.ipynb` and run. It will:
   - Load artifacts & inputs
   - Simulate corners → **p_over/p_under/p_push**, fair odds, EV
   - Convert **goals** O/U odds to probabilities; compute EV
   - Price **parlay** via Gaussian copula (ρ = `rho_copula`, default 0.2)
   - Apply **half-Kelly** by default and **cap total stakes ≤ bankroll**
   - Save `data/matchday_quotes.csv`

---

## In Progress

- Adding goal-line O/U ability to daily pricer
- Implementing further parlay logic to determine if EV > 0 of a correlated parlay
