We are replicating results from "Transformers Represent Belief State Geometry in their Residual Stream" with the Mess3 HMM. The overall workflow is:
1) Define and validate the HMM.
2) Generate synthetic sequences from the HMM to create training data.
3) Pre-train a transformer to predict the next observation in a sequence.
4) Train a linear probe on residual stream activations to recover the Mess3 belief geometry.

Progress so far:
- Implemented the Mess3 HMM.
- Added tests that validate transition dynamics and observation sampling.
- Added the TransformerLens-based transformer model and tests for activation shapes.
- Implemented transformer training in `train_transformer.py`.
- Improved logging for training runs.
- Added KL divergence vs optimal belief state.

Training experiments in progress:
- Optimizer: AdamW vs SGD.
- Sequence length.
- Training steps per sequence.

Next steps:
- Train the transformer on generated sequences.
- Extract residual stream activations and fit a linear probe to evaluate whether the belief geometry is recovered.

