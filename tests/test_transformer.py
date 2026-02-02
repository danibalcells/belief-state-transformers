from __future__ import annotations

import unittest

import torch

from HMM import Mess3
from transformer import BeliefStateTransformer


class TestBeliefStateTransformer(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        torch.manual_seed(0)
        hmm = Mess3()
        model = BeliefStateTransformer.from_paper_config(vocab_size=hmm.vocab_size)
        batch_size = 3
        seq_len = 5
        tokens, _ = hmm.generate_batch(batch_size=batch_size, seq_len=seq_len)

        logits = model.forward_tokens(tokens)

        self.assertEqual(logits.shape, (batch_size, seq_len, hmm.vocab_size))
        self.assertTrue(logits.dtype.is_floating_point)

    def test_forward_with_residuals_shapes(self) -> None:
        torch.manual_seed(0)
        hmm = Mess3()
        model = BeliefStateTransformer.from_paper_config(vocab_size=hmm.vocab_size)
        batch_size = 2
        seq_len = 4
        tokens, _ = hmm.generate_batch(batch_size=batch_size, seq_len=seq_len)

        logits, activations = model.forward_with_residuals(tokens)

        self.assertEqual(logits.shape, (batch_size, seq_len, hmm.vocab_size))
        self.assertEqual(
            activations.shape,
            (model.cfg.n_layers, batch_size, seq_len, model.cfg.d_model),
        )
        self.assertTrue(activations.dtype.is_floating_point)


if __name__ == "__main__":
    unittest.main()
