from __future__ import annotations

import unittest

import torch
from einops import rearrange

from HMM import Mess3


class TestMess3(unittest.TestCase):
    def test_generate_batch_shapes(self) -> None:
        hmm = Mess3()
        batch_size = 4
        seq_len = 7
        seq, hidden = hmm.generate_batch(batch_size=batch_size, seq_len=seq_len)
        self.assertEqual(seq.shape, (batch_size, seq_len))
        self.assertEqual(hidden.shape, (batch_size, seq_len + 1))
        self.assertEqual(seq.dtype, torch.long)
        self.assertEqual(hidden.dtype, torch.long)

    def test_empirical_transition_probabilities(self) -> None:
        torch.manual_seed(0)
        hmm = Mess3()

        batch_size = 50
        seq_len = 10_000
        seq, hidden = hmm.generate_batch(batch_size=batch_size, seq_len=seq_len)

        curr = hidden[:, :-1]
        nxt = hidden[:, 1:]
        x = seq

        idx = curr * 9 + x * 3 + nxt
        counts = torch.bincount(rearrange(idx, "b t -> (b t)"), minlength=27).to(torch.float64)
        counts = rearrange(counts, "(i x j) -> i x j", i=3, x=3, j=3)

        empirical = counts / counts.sum(dim=(1, 2), keepdim=True)
        expected = rearrange(hmm._t_x, "x i j -> i x j").to(torch.float64)

        max_abs_err = float((empirical - expected).abs().max().item())
        print("\nEmpirical transition probabilities (i, x, j):")
        print(empirical)
        print("\nExpected transition probabilities (i, x, j):")
        print(expected)
        print(f"\nMax abs error: {max_abs_err:.6f}")
        self.assertLess(max_abs_err, 1e-2)


if __name__ == "__main__":
    hmm = Mess3()
    seq, hidden = hmm.generate_batch(batch_size=2, seq_len=10)
    print("seq:")
    print(seq)
    print("hidden:")
    print(hidden)
    unittest.main()

