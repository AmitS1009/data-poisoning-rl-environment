# Three of our training feeds are lying to us

We retrain the vibration triage model on data that comes in from four collection feeds.
Three of them started producing models that are noticeably worse on our own acceptance
set, and it happened without the feed volume or the class balance changing. Our working
theory is that somebody has been editing the training rows before we see them. We do not
have clean copies, and the row ordering was shuffled on ingest, so there is nothing to
diff against.

Everything is in `/app/task_inputs/corpora/`. Each corpus directory holds `train_x.npy`
(float32, one 72-bin spectrum per row), `train_y.npy` (int64 class id per training row),
`val_x.npy` / `val_y.npy` (a small clean acceptance split we trust), `test_x.npy` (the
rows we want scored, labels withheld) and `meta.json`. `alpha_feed`, `beta_feed` and
`gamma_feed` are the suspect ones. `probe_corpus` is a fourth feed that never left our
network, so it is clean.

## First, show me the attack

Before I believe the tampering theory I want to see somebody do it to us on purpose.
Poison `probe_corpus` and write the result to `/app/results/attack/`, as
`poisoned_train_x.npy` and `poisoned_train_y.npy`, same shapes and dtypes as the
originals.

The budget is deliberately tight, because the claim is that a small edit does real
damage:

- at most **8%** of the training rows may differ from the originals at all
- on any row you do change, no feature may move by more than **0.75** in absolute value
- labels may only change on rows you have already touched or are touching

It has to be a real poisoning attack, one that measurably costs the victim model
accuracy on rows it has never seen. Shuffling labels at random will not do it and we
will be able to tell.

The victim is fixed and you have it. `/app/task_inputs/reference_model.py` is the exact
architecture, training schedule and scoring convention we grade with — a
72 → 128 → 64 → 5 MLP, dropout 0.2, Adam at lr 0.01 with weight decay 5e-4, 220 full-batch
epochs, averaged over seeds 11/12/13, accuracy read at the best-validation epoch. Measure
yourself with it rather than with a loop of your own:

```
python /app/task_inputs/reference_model.py --corpus probe_corpus \
    --poisoned-x /app/results/attack/poisoned_train_x.npy \
    --poisoned-y /app/results/attack/poisoned_train_y.npy
```

It scores on the validation split, because that is the labels you have. We score the same
way on the withheld rows. Validation is the smaller of the two samples, so give yourself
margin instead of stopping the moment you touch the bar.

## Then the actual job

For each of the three suspect corpora, get test accuracy back up — and give me the
diagnosis, not just the predictions. Per corpus, write:

- `/app/results/defense/<corpus>/weights.npy` — one non-negative float32 per **training**
  row, in row order: how much you trust that row. This is the deliverable I care most
  about, because it is what tells us which rows to go back and audit.
- `/app/results/defense/<corpus>/pred.npy` — one int64 class id per **test** row, in row
  order.
- `/app/results/defense/<corpus>/val_pred.npy` — the same, for the **validation** rows.
  You already predict these to report your own accuracy; shipping them is what lets us
  check the numbers you report against the model that produced them.
- `/app/results/defense/<corpus>/model.pt` — the trained model, saved on the GPU without
  moving it to CPU first, as
  `torch.save({"state_dict": model.state_dict(), "logits": logits}, path)`, where `logits`
  is the `(n_test, n_classes)` tensor your predictions came from. We check that the
  checkpoint still runs a CUDA backward pass and that its logits agree with the `pred.npy`
  you shipped.

The three feeds were not tampered with the same way. A trick that works on one of them
will not work on all three.

## And tell me which part is doing the work

On `alpha_feed`, rerun your method with the full objective and with each of its
regularization terms dropped in turn — at least four variants, one of them named `full` —
writing `/app/results/ablation/<variant>/pred.npy` and
`/app/results/ablation/<variant>/val_pred.npy` for each.

## Wrap-up

`/app/results/metrics.json`:

```json
{"attack": {"victim_val_acc_clean": 0.0, "victim_val_acc_poisoned": 0.0, "rows_modified": 0},
 "defense": {"alpha_feed": {"defense_val_acc": 0.0, "baseline_val_acc": 0.0},
             "beta_feed": {}, "gamma_feed": {}},
 "ablation": {"full": 0.0}}
```

`ablation` keys must match the variant directory names exactly.

Also leave a few paragraphs in `/app/results/report.md`, naming each of the three corpora,
covering the attack, how you did against an unweighted baseline and at least one other
defense you consider a fair comparison, the ablation, and what your weights say each feed
was actually doing to us.

## How we will judge it, so nothing is a surprise

- Mean accuracy on the withheld rows across the three corpora: **at least 0.80**, and
  **at least 0.02 above** an unweighted model we train ourselves on the same poisoned
  corpus and split.
- Your attack has to cost that same reference model **at least 0.05** accuracy on
  `probe_corpus`, within the budget above.
- The weights have to actually find the tampering. We rank the training rows by the
  weight you gave them: the chance that a tampered row outranks a clean one must be
  **at most 0.47** in **every** corpus. Uniform weights, or any constant rescaling of
  them, score exactly 0.5 and fail. Only pushing the tampered rows down relative to the
  rest moves it.
- The weights also have to leave a usable training set. We measure that as effective
  sample size, `(sum w)^2 / sum w^2` — it is unchanged if you rescale every weight, and
  handing a discarded row a token epsilon does not inflate it. It must be at least
  **25%** of the training rows, and at least **10%** again when restricted to the
  untampered ones. Zeroing almost everything and training on a handful of rows is not a
  diagnosis.
- The ablation has to be real: at least three of the variants must produce different
  predictions, and every accuracy in `metrics.json` must land within **0.05** of what we
  recompute from the `val_pred.npy` files you actually shipped. `report.md` needs at least 600
  characters of real write-up.

One GPU, two hours, no network, and the withheld labels stay withheld.
