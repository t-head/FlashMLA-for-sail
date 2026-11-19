import unittest

import torch

from flash_mla import flash_mla_sparse_fwd


class TestHS64LargeKV(unittest.TestCase):
    @torch.inference_mode()
    def test_bf16_row_offsets_across_64_gib(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
            self.skipTest("requires SM89")
        if torch.cuda.mem_get_info()[0] < (65 << 30):
            self.skipTest("requires 65 GiB of free device memory")

        for width in (512, 576):
            with self.subTest(width=width):
                self.check_boundary(width)

    def check_boundary(self, width):
        row_bytes = 2 * width
        first_high = ((1 << 36) + row_bytes - 1) // row_bytes
        rows = [first_high - 1, first_high, first_high + 1]
        kv = torch.empty((first_high + 2, 1, width), device="cuda", dtype=torch.bfloat16)
        # Any wrapped read lands in these rows. The three queried rows are distinct.
        kv[:3].fill_(5)
        values = [16, 32, 64]
        for row, value in zip(rows, values):
            kv[row].fill_(value)
        q = kv.new_zeros((len(rows) + 1, 64, width))
        indices = torch.tensor(rows, device=q.device, dtype=torch.int32)
        indices = indices[:, None, None].expand(-1, 1, 2048)
        # Each XOR4 lane pair must exchange offsets on opposite sides of 64 GiB.
        mixed = torch.tensor(rows[:2], device=q.device, dtype=torch.int32)
        mixed = mixed.repeat_interleave(32).repeat(32)
        mixed[-64:] = -1
        indices = torch.cat((indices, mixed.view(1, 1, -1)))
        out, _, _ = flash_mla_sparse_fwd(q, kv, indices, width**-0.5)
        expected = out.new_tensor([*values, 24])[:, None, None].expand_as(out)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
