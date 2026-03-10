# autoresearch (x-transformers edition)

This is an experiment to have the LLM do its own research on character-level
language modeling using [x-transformers](https://github.com/lucidrains/x-transformers).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `AGENTS.md` — the experiment protocol (hardware, parameters, loop rules).
   - `train.py` — the file you modify. Model config, optimizer, training loop.
   - `docs/adjustable_params.md` — x-transformers parameter reference.
4. **Verify data exists**: Check that `./x-transformers/data/enwik8.gz` exists. If not, download it.
5. **Verify dependencies**: Run `python -c "from x_transformers import TransformerWrapper"` to check. If missing deps, install:
   ```bash
   pip install loguru einx ema-pytorch adam-atan2-pytorch
   # Optional FP8:
   pip install --no-build-isolation transformer_engine[pytorch]
   ```
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). Launch it as:

```bash
python train.py                       # BF16 (default)
USE_FP8=1 python train.py             # FP8 via Transformer Engine (if installed)
```

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture parameters, x-transformers Decoder options, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify files inside `x-transformers/`. The library is read-only reference.
- Break the output format (the `---` summary block at the end must remain parseable).

**The goal is simple: get the lowest val_bpc.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a hard constraint. OOM = crash. Be conservative with batch sizes and model dimensions.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpc improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpc improvement from deleting code? Definitely keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpc:          1.234567
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     12345.6
mfu_percent:      15.00
total_tokens_M:   200.0
num_steps:        500
num_params_M:     19.5
depth:            6
precision:        BF16
```

You can extract the key metric from the log file:

```
grep "^val_bpc:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpc	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpc achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpc	memory_gb	status	description
a1b2c3d	1.234567	8.5	keep	baseline (dim=512 depth=6 heads=8 bf16)
b2c3d4e	1.220000	8.6	keep	add ff_glu=True ff_swish=True
c3d4e5f	1.250000	8.5	discard	reduce depth to 4
d4e5f6g	0.000000	0.0	crash	dim=1024 OOM
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar10`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpc:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_bpc improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpc is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read `docs/adjustable_params.md` and the x-transformers source for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
