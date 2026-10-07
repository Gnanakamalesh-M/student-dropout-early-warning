# Future work

Ordered by what would most change the conclusions, not by what would be most fun
to build. Each entry says what is wrong now and what evidence would settle it.

---

## 1. The label is the weakest part of the whole system

The target is formal de-registration. A student who stops engaging in week 4 and
never files paperwork is labelled **negative** at every checkpoint, and the model
is trained to call them safe.

This is not a minor annotation issue — it is the most likely reason the ceiling is
where it is. It also probably biases the fairness audit: administrative
withdrawal is an act with a cost (knowing the process, having the confidence to
use it), so under-counting is unlikely to be uniform across groups, and every
subgroup rate in `ETHICS.md` inherits that bias in a direction the audit cannot
measure.

**What would settle it:** a second label built from sustained disengagement — no
activity for *k* weeks with assessments still outstanding — and the two models
compared. If the disengagement model is materially better, the current target is
the bottleneck rather than the features. This is the single highest-value
experiment left, and it is a data question, not a modelling one.

## 2. Per-checkpoint operating points

Measured in `MONITORING.md`: at one global cutoff, the alert rate is **9.6% at
checkpoint 30 and 2.5% at checkpoint 180** against a nominal 5% budget. Late-term
rows score systematically lower, so one threshold produces very uneven workload.

If staffing is allocated per checkpoint — which is how a real support team works —
the threshold should be derived per checkpoint too. That is a change to the
operating point, not the model, and it needs someone who knows how the team is
actually rostered. Deliberately not guessed here.

## 3. Intersectional fairness, which this cohort cannot support

Each protected attribute is audited marginally. A disparity affecting a
*combination* would not appear. At a 3% positive rate the cells are far too small
— `age_band 55<=` alone already has 13 positives and is reported as unassessable.

**What would settle it:** more data, or a deliberately pooled analysis across
institutions. Reporting intersectional point estimates on this cohort would
manufacture exactly the kind of finding the audit is built to refuse.

## 4. Whether the interventions do anything

The system recommends support actions. **Nothing here establishes that any of them
help.** They are plausible and non-punitive; that is the whole claim. An
early-warning system whose interventions are untested is a triage tool, not a
treatment.

**What would settle it:** a randomised or staggered rollout with the outcome
measured. That is a programme evaluation with ethical review, not a feature.
Until it exists, the recommendations should keep being framed as suggestions for a
human to consider.

## 5. Concept drift, as opposed to distribution drift

`monitoring/drift.py` measures whether inputs moved. It cannot measure whether the
*relationship* between features and withdrawal moved — that needs resolved labels
and a second fitted model to compare against. That is a retraining experiment
rather than a monitor, and it is why the module escalates to `requires_review` and
has no `requires_retraining`.

## 6. Calibration per subgroup, properly tested

The fairness audit found the largest calibration gap on `disability Y`: predicted
3.32% against an observed 4.65%, i.e. the model **understates** risk for the group
with the highest base rate. It is small, it changes nobody's flag at a 5% budget,
and **it has not been significance-tested**.

**What would settle it:** the same cluster bootstrap used for the headline
intervals, applied to the per-subgroup calibration gap. Cheap to do and currently
missing — this is the most obvious hole in the fairness work.

## 7. ~~Verify the containers somewhere the author can see~~ — done, mostly

**Closed locally.** Docker Desktop was installed and the stack built and run end
to end: all three services `healthy`, and a real SHAP-explained scoring request
served from the container (probability 0.3333, `critical` band, 6 risk factors,
`is_calibrated: true`), with cohort figures matching the non-container run
exactly. Building it for the first time found four bugs, all of which produced a
healthy-looking failure — see `docs/DEPLOYMENT.md`.

**What remains:** CI still runs the `containers` job without a trained model,
because `models/` is not committed. So the automated gate proves wiring, not
scoring, and a path regression of the kind found here would be invisible to it —
CI expects `degraded` and cannot distinguish that from a broken path.

**What would settle that part:** a small committed fixture model, scored through
the containerised API in CI, with an assertion on `model_loaded: true` rather
than on the endpoint merely answering. That is the remaining work; it is smaller
than it was.

## 8. Things deliberately left out, with reasons

- **Deep learning on raw clickstream sequences.** Plausibly better, and would cost
  the explanation layer that makes this usable by a counsellor. Not a good trade
  for this application until the label problem (#1) is fixed — there is little
  point adding capacity to fit a target that is partly wrong.
- **Survival analysis.** Arguably the more natural framing (time-to-withdrawal
  with censoring) than six independent binary checkpoints. Worth an ADR of its
  own; the checkpoint framing was chosen for operational legibility, and that
  choice deserves re-examination rather than defence.
- **Kubernetes manifests, TLS termination, rate limits, retention policy.** Each
  would be unverifiable both locally *and* in CI, which is a different category
  from the compose file. Guesswork presented as deployment config is worse than
  its absence.
- **Automatic retraining.** Refused on evidence, not caution: a full
  presentation's worth of major feature drift coexisted with stable, calibrated
  performance. See `MONITORING.md`.
