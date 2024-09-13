import torch

def calculate_diff(a, b):
    return torch.abs(a-b)

def max_diff(a, b):
    return torch.max(calculate_diff(a, b))

def average_diff(a, b):
    return torch.mean(calculate_diff(a, b))

def report(a, b, compare_top=False):
    maximum_diff = max_diff(a, b)
    avg_diff = average_diff(a, b)
    if compare_top:
        worst_diff = 0
        _, top_indices = torch.topk(a.flatten(), 20)
        for idx in top_indices:
            diff = calculate_diff(a.flatten()[idx], b.flatten()[idx])
            if diff > worst_diff:
                worst_diff = diff

    return (
        f"Same? {compare_same(a, b)} {compare_multiple_tolerances(a, b)}\n",
        f"Max diff:\t{maximum_diff:.4e}\n",
        f"Average diff:\t{avg_diff:.4e}",
        f"\nTop 20 > diff:\t{worst_diff:.4e} [{worst_diff:.8f}%pt]" if compare_top else ""
    )

def compare_same(a, b):
    return f"{condition_string(torch.equal(a, b))} (exact)"
    
def compare_multiple_tolerances(a, b):
    return f"{condition_string(torch.allclose(a, b, atol=1e-3))}{condition_string(torch.allclose(a, b, atol=1e-5))}{condition_string(torch.allclose(a, b, atol=1e-9))}{condition_string(torch.allclose(a, b, atol=1e-12))} (w tolereances)"

def condition_string(condition):
    if condition:
        return "✅"
    else:
        return "❌"

def compare_top_k(a, b, k: int, dim: int):
    top_vals_a, top_indices_a = torch.topk(a, k, dim=dim)
    top_vals_b, top_indices_b = torch.topk(b, k, dim=dim)
    # max_diff_top_vals = max_diff(top_vals_a, top_vals_b)
    # average_diff_top_vals = average_diff(top_vals_a, top_vals_b)
    are_same = torch.equal(top_indices_a, top_indices_b)
    return are_same
    