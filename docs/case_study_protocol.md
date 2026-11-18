# Case-study protocol: what was fixed, and when

This file records the protocol of the 2021 retailer backtest (Section 7 of the article)
as it was fixed before the backtests were run, and every change made to it afterwards.
It is an English rendering of the authors' dated working record, which is kept in
Chinese in the private development history of this project. The dates below are those
of that history; this file itself was added to the public repository in November 2026,
so its own timestamp does not certify them.

## Timeline

| Date (2026, UTC+8) | Event |
|---|---|
| 16 Oct, 14:20 | Acceptance criteria, methods, budget and analysis fixed (Sections 1-5 below). |
| 16 Oct, afternoon | Code tests and pilot runs on June and July 2021 (one small retailer, one seed; one full-budget NSGA-II run settled on July 2021) and baseline diagnostics on January and July 2021. |
| 16 Oct, 17:10 | Amendment A (baseline rounding, initial population, training data). Criteria unchanged. |
| 17 Oct | Three descriptive additions registered before any backtest (Section 6). |
| 18 Oct | Ranking check failed twice; Amendment B (surrogate readout and supervision). Criterion unchanged. |
| 19 Oct, 04:48-05:36 | Ranking check passed for all three retailers; unit costs measured; backtests started. |
| 20 Oct, 16:02 | All 1,458 backtest runs complete; analysis run once on the full set. |

## 1. Design

- Retailers: three archetypes (S, M, L), each at rho in {0.05, 0.1, 0.3}: nine settings.
- Months: the nine months of 2021 with complete records (1, 3, 4, 6-11), three seeds each:
  27 cases per setting. Each month is decided with the data available two months ahead
  and settled on that month's actual generation and load.
- Objectives per setting are those that pass a conflict test fixed beforehand: on January
  and July 2021, three seeds, a pair of indicators is accepted if at least four of the six
  cases give a front of at least ten distinct points whose extremes differ by at least 5%
  of each indicator's range. A setting with no accepted pair is not backtested.
  (Outcome: profit versus worst fulfilment in all nine settings.)

## 2. Precondition on the surrogate

Trained only on months before 2021. On converged NSGA-II populations of a 2021 instance,
five seeds: the local Spearman correlation of every indicator must be positive
(one-sided Wilcoxon, p < 0.05, which requires all five seeds positive) and its mean at
least 0.5. Checked on one retailer first, then on the others. A retailer for which exact
evaluation does not cost more than two predictions is not backtested.

## 3. Methods and budget

- Methods, same instance and seed: contract-proportional baseline; last month's selected
  allocation reused; NSGA-II; SA-NSGA-II with the learned surrogate; SA-NSGA-II with
  random predictions; SA-NSGA-II screening with an exact settlement on k scenarios,
  k = round(S x c_sur / c_exact), at least 1, so that its screening cost matches the
  surrogate's.
- SA settings are the published ones, not retuned: retention 0.5, 10 warm-up
  generations, a full evaluation every 20 generations, random exploration 0.1,
  screened-out offspring discarded.
- Cost is counted, not timed: exact evaluations, predictions and few-scenario
  settlements are charged at unit costs measured single-threaded on an idle machine.
  Every method spends B = 10,000 exact evaluations' worth of cost.

## 4. Computational benefit

- Hypervolume on the scenario indicators, normalised per instance by the union of all
  methods' final fronts, reference point 1.1.
- Target: 99% of NSGA-II's hypervolume at budget B. Speed-up of method X: cost NSGA-II
  needs to reach the target divided by the cost X needs; a method that does not reach it
  within B scores just below one.
- Criteria, paired over the 27 cases, one-sided Wilcoxon, p < 0.05:
  (a) the surrogate's speed-up exceeds one;
  (b) it exceeds that of random screening;
  (d) it exceeds that of few-scenario screening.

## 5. Realized benefit

- Selection rule, applied before the month is known: among the allocations on the final
  front whose mean worst fulfilment is at least the baseline's, the one with the highest
  mean profit; if none qualifies, the one with the highest mean worst fulfilment.
- Criterion (c): the selection earns more than the baseline without a lower worst
  fulfilment in at least two thirds of the 27 cases, and its profit gain is positive
  (one-sided Wilcoxon, p < 0.05).
- Holm correction over all 36 tests (nine settings, four criteria).
- The claim that the surrogate provides a concrete optimization benefit on the real case
  is made only if (a), (b), (c) and (d) all hold; otherwise only what holds is reported.
- A step that fails stops the sequence and is reported; criteria are not revised.

## 6. Additions registered on 17 October, before any backtest

Descriptive only, outside the 36-test family.
- Realized profit of SA-NSGA-II against NSGA-II at 25%, 50%, 75% and 100% of the budget
  (one-sided Wilcoxon with its own Holm family).
- Two reference runs per case, entering no criterion: NSGA-II on the mean scenario, and
  NSGA-II given the month's actual records (a hindsight bound). After a code test showed
  the hindsight search too weak at budget B, its budget was set to 5B before any backtest.

## Amendments

**A. Baseline, starting point and training data (16 October, after the pilot runs,
before any reported backtest).** The pilot runs showed two things. Rounding each cell of
the contract-proportional allocation to the 0.01% resolution of the wheeling contract
sets the smallest buyers' shares to zero, so that baseline was not an allocation a
retailer would file; and a search started from random allocations did not reach the
neighbourhood of the baseline within the budget. Three settings were therefore changed:
(i) the baseline is the contract-proportional allocation written at contract precision,
so that every buyer keeps a positive expected volume; (ii) every search starts from a
population of the baseline plus 99 uniformly drawn allocations; (iii) the surrogate's
training data include the baseline and perturbations of it. The change makes the
baseline stronger. No criterion, test, selection rule or budget was changed, and the
runs made before the amendment are not among the reported results.

**B. Surrogate readout (18 October, before any backtest).** The model with the
mean-pooled readout of the main experiments failed the ranking check of Section 2 on
worst fulfilment, and so did a soft-minimum readout trained on the graph-level target.
A soft-minimum readout with per-buyer supervision passed for all three retailers. The
criterion was not changed; all three models are reported in the supplementary material.

## Outcome

Reported in Section 7 and Table 12 of the article: (a) holds in three of nine settings,
(b) and (d) in none, (c) in one.
