import unittest

import torch

from sam2_trt.real_rope import apply_rotary_enc_real


def complex_reference(query, key, frequencies, repeat):
    query_complex = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
    key_complex = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
    frequencies = frequencies.view(1, 1, *frequencies.shape)
    query_output = torch.view_as_real(query_complex * frequencies).flatten(3)
    if repeat:
        repetitions = key.shape[-2] // query.shape[-2]
        frequencies = (
            frequencies.unsqueeze(2)
            .expand(-1, -1, repetitions, -1, -1)
            .flatten(2, 3)
        )
    key_output = torch.view_as_real(key_complex * frequencies).flatten(3)
    return query_output.type_as(query), key_output.type_as(key)


class RealRopeTest(unittest.TestCase):
    def test_matches_complex_reference(self):
        torch.manual_seed(7)
        query = torch.randn(2, 4, 16, 32)
        key = torch.randn(2, 4, 48, 32)
        phase = torch.randn(16, 16)
        frequencies = torch.polar(torch.ones_like(phase), phase)
        expected_query, expected_key = complex_reference(query, key, frequencies, True)
        actual_query, actual_key = apply_rotary_enc_real(
            query, key, frequencies.real, frequencies.imag, True
        )
        torch.testing.assert_close(actual_query, expected_query, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(actual_key, expected_key, rtol=1e-6, atol=1e-6)

    def test_matches_without_key_repetition(self):
        torch.manual_seed(11)
        query = torch.randn(1, 2, 9, 8)
        key = torch.randn(1, 2, 9, 8)
        phase = torch.randn(9, 4)
        frequencies = torch.polar(torch.ones_like(phase), phase)
        expected = complex_reference(query, key, frequencies, False)
        actual = apply_rotary_enc_real(
            query, key, frequencies.real, frequencies.imag, False
        )
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
