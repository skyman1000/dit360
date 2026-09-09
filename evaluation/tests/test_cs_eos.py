import unittest
from types import SimpleNamespace
import torch
import numpy as np
from evaluation.metrics.clip_score import encode_text, cosine_score
from evaluation.prepare_mp3d import select_per_scene


class Tokenizer:
    eos_token_id = 99
    def __call__(self, text, **kwargs):
        return {'input_ids': [5] * int(text)}
    def num_special_tokens_to_add(self):
        return 2
    def decode(self, ids, **kwargs):
        return 'decoded'


class Processor:
    tokenizer = Tokenizer()
    broken = False
    def __call__(self, text, **kwargs):
        assert kwargs['truncation'] is True
        tokens = [98] + [5] * int(text[0]) + [99]
        limit = kwargs['max_length']
        if len(tokens) > limit:
            tokens = tokens[:limit]
            if not self.broken:
                tokens[-1] = 99
        return {'input_ids': torch.tensor([tokens]), 'attention_mask': torch.ones(1, len(tokens), dtype=torch.long)}


class EOSTests(unittest.TestCase):
    def test_short_boundary_and_long_text(self):
        for count in (2, 75, 76, 120):
            encoded, info = encode_text(Processor(), str(count), 77)
            self.assertEqual(encoded['input_ids'][0, -1], 99)
            self.assertEqual(info['truncated'], count + 2 > 77)

    def test_missing_eos_fails_before_feature_extraction(self):
        processor = Processor()
        processor.broken = True
        with self.assertRaisesRegex(ValueError, 'EOS'):
            encode_text(processor, '120', 77)

    def test_cosine_scale_and_negative_clamp(self):
        a = torch.tensor([[2., 0.], [1., 0.]])
        b = torch.tensor([[3., 0.], [-1., 0.]])
        torch.testing.assert_close(cosine_score(a, b), torch.tensor([100., 0.]))

    def test_sampling_covers_buildings_and_is_reproducible(self):
        entries = np.array([[f'{s}/view{i}'] for s in ('a', 'b', 'c') for i in range(8)])
        selected = select_per_scene(entries, 2, 0)
        self.assertEqual(len(selected), 6)
        self.assertEqual({r[0].split('/')[0] for r in selected}, {'a', 'b', 'c'})
        np.testing.assert_array_equal(selected, select_per_scene(entries, 2, 0))


if __name__ == '__main__':
    unittest.main()
