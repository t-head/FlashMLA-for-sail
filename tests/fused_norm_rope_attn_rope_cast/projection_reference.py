import torch

from tile_kernels.torch import cast_back
from tile_kernels.utils import align


def get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float and x.dim() in (2, 3)
    ue8m0_tensor = (x.view(torch.int) >> 23).to(torch.uint8)
    mn, k = x.shape[-2], x.shape[-1]
    remove_dim = False
    if x.dim() == 2:
        x, remove_dim = x.unsqueeze(0), True
    b = x.shape[0]
    aligned_mn = align(mn, 16 // 4)
    aligned_k = align(k, 4)
    padded = torch.zeros((b, aligned_mn, aligned_k), device=x.device, dtype=torch.uint8)
    padded[:, :mn, :k] = ue8m0_tensor
    padded = padded.view(-1).view(dtype=torch.int).view(b, aligned_mn, aligned_k // 4)
    transposed = torch.zeros((b, aligned_k // 4, aligned_mn), device=x.device, dtype=torch.int).mT
    transposed[:, :, :] = padded
    aligned_x = transposed[:, :mn, :]
    return aligned_x.squeeze(0) if remove_dim else aligned_x


def fp8_gemm_nt_reference(a, b, out, gran):
    lhs = cast_back(a, 'fp32', (1, gran))
    rhs = cast_back(b, 'fp32', (1, gran))
    out.copy_((lhs @ rhs.T).to(out.dtype))


def fp8_einsum_reference(a, b, out, gran):
    groups = a[0].size(1)
    lhs = torch.stack([
        cast_back((a[0][:, group, :], a[1][:, group, :]), 'fp32', (1, gran))
        for group in range(groups)
    ], dim=1)
    rhs = torch.stack([
        cast_back((b[0][group], b[1][group]), 'fp32', (1, gran))
        for group in range(groups)
    ], dim=0)
    out.copy_(torch.einsum('bhr,hdr->bhd', lhs, rhs).to(out.dtype))
