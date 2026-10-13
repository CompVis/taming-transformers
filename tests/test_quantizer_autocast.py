import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from taming.modules.vqvae.quantize import VectorQuantizer, VectorQuantizer2


def cpu_autocast_dtypes():
    if not hasattr(torch, "autocast"):
        return ()
    supported = []
    for dtype in (torch.bfloat16, torch.float16):
        try:
            with torch.autocast("cpu", dtype=dtype):
                product = torch.ones(1, 2).mm(torch.ones(2, 1))
            if product.dtype == dtype:
                supported.append(dtype)
        except (RuntimeError, TypeError, ValueError):
            continue
    return tuple(supported)


class QuantizerAutocastTests(unittest.TestCase):
    def autocast_dtypes(self):
        supported = cpu_autocast_dtypes()
        if not supported:
            self.skipTest("No supported CPU autocast dtype")
        return supported

    def assert_tensor_close(self, actual, expected, atol=1e-6, rtol=1e-6):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.allclose(actual, expected, atol=atol, rtol=rtol),
                        "{} != {}".format(actual, expected))

    def quantizer(self, weight, **options):
        quantizer = VectorQuantizer2(weight.shape[0], weight.shape[1],
                                     options.pop("beta", 0.25), **options)
        quantizer = quantizer.to(dtype=weight.dtype)
        with torch.no_grad():
            quantizer.embedding.weight.copy_(weight)
        return quantizer

    def pair(self, autocast_dtype, dtype=torch.float32):
        value = 1.0625 if autocast_dtype == torch.bfloat16 else 1.015625
        return torch.tensor([[1.0, 1.0], [value, value]], dtype=dtype), value

    def direct_indices(self, latent, weight):
        flattened = latent.detach().permute(0, 2, 3, 1).reshape(-1, weight.shape[1]).double()
        distances = ((flattened[:, None, :] - weight.detach().double()[None, :, :]) ** 2).sum(2)
        return distances.argmin(1)

    def test_exact_nearest_code_with_float32_inputs_under_autocast(self):
        for dtype in self.autocast_dtypes():
            for training in (True, False):
                with self.subTest(autocast=dtype, training=training):
                    weight, value = self.pair(dtype)
                    quantizer = self.quantizer(weight).train(training)
                    latent = torch.full((1, 2, 1, 1), value)
                    reference = quantizer(latent)
                    self.assertEqual(reference[2][2].tolist(), [1])
                    with torch.autocast("cpu", dtype=dtype):
                        quantized, loss, info = quantizer(latent)
                    self.assertEqual(info[2].tolist(), [1])
                    self.assert_tensor_close(quantized, latent)
                    self.assertEqual(loss.item(), 0.0)

    def test_spatial_noncontiguous_inputs_and_index_shapes(self):
        for dtype in self.autocast_dtypes():
            weight, _ = self.pair(dtype)
            weight = torch.cat((weight, torch.zeros(1, 2)))
            codes = torch.tensor([0, 1, 2, 1, 0, 2, 2, 1, 0, 0, 1, 2])
            latent = weight[codes].reshape(2, 2, 3, 2).permute(0, 3, 2, 1)
            self.assertFalse(latent.is_contiguous())
            expected = self.direct_indices(latent, weight)
            for sane in (False, True):
                with self.subTest(autocast=dtype, sane=sane):
                    quantizer = self.quantizer(weight, sane_index_shape=sane)
                    with torch.autocast("cpu", dtype=dtype):
                        quantized, loss, info = quantizer(latent)
                    self.assertTrue(torch.equal(info[2].reshape(-1), expected))
                    self.assertEqual(info[2].shape, latent.shape[:1] + latent.shape[2:] if sane else (12,))
                    self.assert_tensor_close(quantized, latent)
                    self.assertEqual(loss.item(), 0.0)
                    self.assertIsNone(info[0])
                    self.assertIsNone(info[1])

    def test_latent_codebook_dtypes_and_float64_autocast_control(self):
        for amp_dtype in self.autocast_dtypes():
            for latent_dtype, codebook_dtype in ((torch.float32, torch.float32),
                                                 (amp_dtype, torch.float32),
                                                 (amp_dtype, amp_dtype),
                                                 (torch.float64, torch.float64)):
                with self.subTest(autocast=amp_dtype, latent=latent_dtype, codebook=codebook_dtype):
                    weight, value = self.pair(amp_dtype, codebook_dtype)
                    quantizer = self.quantizer(weight)
                    latent = torch.full((1, 2, 1, 1), value, dtype=latent_dtype)
                    with torch.autocast("cpu", dtype=amp_dtype):
                        quantized, loss, info = quantizer(latent)
                    self.assertEqual(info[2].tolist(), [1])
                    expected_dtype = torch.promote_types(latent_dtype, codebook_dtype)
                    self.assert_tensor_close(quantized, latent.to(dtype=expected_dtype))
                    self.assertEqual(loss.dtype, expected_dtype)
                    self.assertEqual(loss.item(), 0.0)

    def test_legacy_beta_loss_encoder_and_embedding_gradients(self):
        for dtype in self.autocast_dtypes():
            for legacy in (False, True):
                for beta in (0.25, 1.0):
                    with self.subTest(autocast=dtype, legacy=legacy, beta=beta):
                        weight, value = self.pair(dtype)
                        quantizer = self.quantizer(weight, legacy=legacy, beta=beta)
                        offset = 0.0075 if dtype == torch.bfloat16 else 0.0001
                        latent = torch.full((1, 2, 1, 1), value + offset, requires_grad=True)
                        chosen = weight[1].reshape(1, 2, 1, 1)
                        enc_weight, emb_weight = (1.0, beta) if legacy else (beta, 1.0)
                        expected_loss = (enc_weight + emb_weight) * (latent.detach() - chosen).square().mean()
                        expected_latent_grad = torch.ones_like(latent) + 2 * enc_weight * (latent.detach() - chosen) / latent.numel()
                        expected_weight_grad = torch.zeros_like(weight)
                        expected_weight_grad[1] = 2 * emb_weight * (chosen - latent.detach()).reshape(2) / latent.numel()
                        with torch.autocast("cpu", dtype=dtype):
                            quantized, loss, info = quantizer(latent)
                        self.assertEqual(info[2].tolist(), [1])
                        self.assert_tensor_close(loss, expected_loss)
                        (quantized.sum() + loss).backward()
                        self.assert_tensor_close(latent.grad, expected_latent_grad)
                        self.assert_tensor_close(quantizer.embedding.weight.grad, expected_weight_grad)

    def test_remap_unknown_policies_decode_rng_and_state(self):
        with tempfile.TemporaryDirectory() as directory:
            remap = str(Path(directory) / "used.npy")
            np.save(remap, np.array([1, 2], dtype=np.int64))
            for dtype in self.autocast_dtypes():
                weight, _ = self.pair(dtype)
                weight = torch.cat((weight, torch.zeros(1, 2)))
                latent = weight[torch.tensor([0, 1, 2])].reshape(1, 1, 3, 2).permute(0, 3, 1, 2)
                for unknown in ("random", "extra", 0):
                    for sane in (False, True):
                        with self.subTest(autocast=dtype, unknown=unknown, sane=sane):
                            quantizer = self.quantizer(weight, remap=remap, unknown_index=unknown,
                                                       sane_index_shape=sane)
                            before = {name: value.clone() for name, value in quantizer.state_dict().items()}
                            torch.manual_seed(23)
                            expected = quantizer.remap_to_used(torch.tensor([[0, 1, 2]]))
                            expected_rng = torch.get_rng_state()
                            torch.manual_seed(23)
                            with torch.autocast("cpu", dtype=dtype):
                                quantized, loss, info = quantizer(latent)
                            self.assertTrue(torch.equal(info[2].reshape(-1), expected.reshape(-1)))
                            self.assertEqual(info[2].shape, (1, 1, 3) if sane else (3, 1))
                            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                            self.assert_tensor_close(quantized, latent)
                            self.assertEqual(loss.item(), 0.0)
                            self.assertEqual(set(before), {"embedding.weight", "used"})
                            for name, value in before.items():
                                self.assertTrue(torch.equal(quantizer.state_dict()[name], value))
                            decoded = quantizer.get_codebook_entry(torch.tensor([0, 1]), (1, 1, 2, 2))
                            self.assert_tensor_close(decoded, weight[1:].reshape(1, 1, 2, 2).permute(0, 3, 1, 2))

    def test_unmixed_fp32_fp64_match_native_legacy_quantizer(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                weight = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, -1.0]], dtype=dtype)
                original = VectorQuantizer(3, 2, 0.25).to(dtype=dtype)
                improved = self.quantizer(weight)
                with torch.no_grad():
                    original.embedding.weight.copy_(weight)
                latent = (weight[torch.tensor([0, 1, 2, 1])] + 0.03).reshape(1, 2, 2, 2).permute(0, 3, 1, 2)
                old_input = latent.clone().requires_grad_()
                new_input = latent.clone().requires_grad_()
                old_output, old_loss, old_info = original(old_input)
                new_output, new_loss, new_info = improved(new_input)
                tolerance = 1e-6 if dtype == torch.float32 else 1e-12
                self.assert_tensor_close(new_output, old_output, atol=tolerance, rtol=tolerance)
                self.assert_tensor_close(new_loss, old_loss, atol=tolerance, rtol=tolerance)
                self.assertTrue(torch.equal(new_info[2].reshape(-1), old_info[2].reshape(-1)))
                (old_output.sum() + old_loss).backward()
                (new_output.sum() + new_loss).backward()
                self.assert_tensor_close(new_input.grad, old_input.grad, atol=tolerance, rtol=tolerance)
                self.assert_tensor_close(improved.embedding.weight.grad, original.embedding.weight.grad,
                                         atol=tolerance, rtol=tolerance)

    def test_empty_inputs_keep_existing_empty_indices_and_nan_loss(self):
        for dtype in (torch.float32, torch.float64):
            for sane in (False, True):
                with self.subTest(dtype=dtype, sane=sane):
                    quantizer = self.quantizer(torch.zeros(2, 2, dtype=dtype), sane_index_shape=sane)
                    latent = torch.empty(0, 2, 1, 1, dtype=dtype)
                    quantized, loss, info = quantizer(latent)
                    self.assertEqual(quantized.shape, latent.shape)
                    self.assertEqual(quantized.dtype, latent.dtype)
                    self.assertEqual(info[2].shape, (0, 1, 1) if sane else (0,))
                    self.assertEqual(info[2].dtype, torch.int64)
                    self.assertTrue(torch.isnan(loss).item())

    def test_interface_options_ties_and_unmodified_state_contract(self):
        for dtype in self.autocast_dtypes():
            weight, value = self.pair(dtype)
            weight[0].copy_(weight[1])
            quantizer = self.quantizer(weight)
            latent = torch.full((1, 2, 1, 1), value)
            self.assertEqual(set(quantizer.state_dict()), {"embedding.weight"})
            with torch.autocast("cpu", dtype=dtype):
                _, _, info = quantizer(latent, temp=1.0, rescale_logits=False, return_logits=False)
            self.assertEqual(info[2].tolist(), [0])
            with self.assertRaises(AssertionError):
                quantizer(latent, temp=2.0)
            with self.assertRaises(AssertionError):
                quantizer(latent, rescale_logits=True)
            with self.assertRaises(AssertionError):
                quantizer(latent, return_logits=True)


if __name__ == "__main__":
    unittest.main()
