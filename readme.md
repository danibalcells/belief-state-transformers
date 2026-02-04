We are replicating results from "Transformers Represent Belief State Geometry in their Residual Stream" with the Mess3 HMM. The overall workflow is:
1) Define and validate the HMM.
2) Generate synthetic sequences from the HMM to create training data.
3) Pre-train a transformer to predict the next observation in a sequence.
4) Train a linear probe on residual stream activations to recover the Mess3 belief geometry.

Progress so far:
- Implemented the Mess3 HMM.
- Added tests that validate transition dynamics and observation sampling.
- Added the TransformerLens-based transformer model and tests for activation shapes.
- Implemented transformer training in `scripts/train_transformer.py`.
- Implemented activation sampling in `scripts/sample_acts.py`.
- Implemented linear probe training in `scripts/train_probe.py`.
- Improved logging for training runs.
- Added KL divergence vs optimal belief state.

Training experiments in progress:
- Optimizer: AdamW vs SGD.
- Sequence length.
- Training steps per sequence.

Implementation notes:
- `scripts/sample_acts.py` loads a trained checkpoint, samples HMM sequences, computes beliefs, and saves flattened activations plus belief targets to `outputs/datasets/<run_id>/dataset.pt`.
- `scripts/train_probe.py` trains `probes.LinearProbe` on the saved dataset with an MSE loss, logs to Weights & Biases, and saves probe weights to `outputs/probes/<run_id>/probe.pt`.
- The dataset file includes `acts`, `states`, `beliefs`, `seq_len`, `resid_stage`, `layers`, and `num_sequences`.

Next steps:
- Train the transformer on generated sequences.
- Run `scripts/sample_acts.py` to build a probe dataset, then `scripts/train_probe.py` to evaluate recovered belief geometry.

