#!/usr/bin/env python3
"""
Minimal test to diagnose Triton uint64 shift and bitwise operation behavior.
Run this on GPU to identify which operations work correctly.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def test_shifts_kernel(
    Input_Ptr,    # int64 [N]
    Output_Ptr,   # int64 [N * 8]  (8 outputs per input)
    N: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return
    
    # Load as int64
    val_i64 = tl.load(Input_Ptr + pid)
    
    # Test 1: raw left shift on int64
    r0 = val_i64 << 2
    tl.store(Output_Ptr + pid * 8 + 0, r0)
    
    # Test 2: cast to uint64 first, then left shift
    val_u64 = val_i64.to(tl.uint64)
    r1 = val_u64 << 2
    tl.store(Output_Ptr + pid * 8 + 1, r1.to(tl.int64))
    
    # Test 3: right shift on int64 (arithmetic?)
    r2 = val_i64 >> 62
    tl.store(Output_Ptr + pid * 8 + 2, r2)
    
    # Test 4: right shift on uint64 (logical?)
    r3 = val_u64 >> 62
    tl.store(Output_Ptr + pid * 8 + 3, r3.to(tl.int64))
    
    # Test 5: right shift + mask on int64
    r4 = (val_i64 >> 62) & 3
    tl.store(Output_Ptr + pid * 8 + 4, r4)
    
    # Test 6: bitwise NOT on int64
    r5 = ~val_i64
    tl.store(Output_Ptr + pid * 8 + 5, r5)
    
    # Test 7: bitwise NOT on uint64
    r6 = ~val_u64
    tl.store(Output_Ptr + pid * 8 + 6, r6.to(tl.int64))
    
    # Test 8: compound shift (lo << k) | (hi >> (64-k))
    # Simulate: lo=val, hi=0, k=2
    lo = val_u64
    hi = tl.zeros([], dtype=tl.uint64)
    new_lo = lo << 2
    new_hi = (hi << 2) | ((lo >> 62) & 3)
    tl.store(Output_Ptr + pid * 8 + 7, new_hi.to(tl.int64))

@triton.jit 
def test_negate_kernel(
    Input_Ptr,   # int64 [3] - p0, p1, p2
    Output_Ptr,  # int64 [3] - result
):
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    p0 = tl.load(Input_Ptr + 0).to(tl.uint64)
    p1 = tl.load(Input_Ptr + 1).to(tl.uint64)
    p2 = tl.load(Input_Ptr + 2).to(tl.uint64)
    
    is_zero = ~(p0 | p1 | p2)
    mask = ~is_zero
    
    n0 = (~p0) & mask
    n1 = (~p1) & mask
    n2 = (~p2) & mask
    
    tl.store(Output_Ptr + 0, n0.to(tl.int64))
    tl.store(Output_Ptr + 1, n1.to(tl.int64))
    tl.store(Output_Ptr + 2, n2.to(tl.int64))


def main():
    device = torch.device("cuda")
    
    # Test values
    test_vals = [
        0,
        4,        # bit 2
        16,       # bit 4
        (1 << 62),  # bit 62 (tests crossing into hi word)
        -((1 << 63) - 1),  # bit pattern 0x8000_0000_0000_0001 (negative as int64)
        -1,  # all 1s (0xFFFF... as int64)
    ]
    
    N = len(test_vals)
    input_data = torch.tensor(test_vals, dtype=torch.int64, device=device)
    output_data = torch.zeros(N * 8, dtype=torch.int64, device=device)
    
    test_shifts_kernel[(N,)](input_data, output_data, N, num_warps=1)
    torch.cuda.synchronize()
    
    out = output_data.cpu()
    
    labels = [
        "int64 << 2",
        "uint64 << 2", 
        "int64 >> 62",
        "uint64 >> 62",
        "(int64>>62)&3",
        "~int64",
        "~uint64",
        "compound_hi",
    ]
    
    print("=" * 100)
    print("Triton Shift/Bitwise Diagnostics")
    print("=" * 100)
    
    for i, val in enumerate(test_vals):
        val_u = val if val >= 0 else val + (1 << 64)
        print(f"\nInput: {val_u:#018x} (int64={val})")
        
        for j, label in enumerate(labels):
            gpu_val = out[i * 8 + j].item()
            gpu_u = gpu_val if gpu_val >= 0 else gpu_val + (1 << 64)
            
            # Compute expected (Python reference, unsigned)
            if j == 0:  # int64 << 2
                expected = (val_u << 2) & ((1 << 64) - 1)
            elif j == 1:  # uint64 << 2
                expected = (val_u << 2) & ((1 << 64) - 1)
            elif j == 2:  # int64 >> 62 (arithmetic)
                if val < 0:
                    expected = (-1) & ((1 << 64) - 1)  # sign extended
                else:
                    expected = val_u >> 62
                # Actually unclear what Triton does - show both
                expected_arith = val >> 62 if val >= 0 else (val >> 62) 
                expected_logic = val_u >> 62
            elif j == 3:  # uint64 >> 62 (logical)
                expected = val_u >> 62
            elif j == 4:  # (int64>>62)&3
                expected = (val_u >> 62) & 3
            elif j == 5:  # ~int64
                expected = (~val) & ((1 << 64) - 1)
            elif j == 6:  # ~uint64
                expected = (val_u ^ ((1 << 64) - 1))
            elif j == 7:  # compound_hi (hi=0, lo=val, k=2)
                expected = (val_u >> 62) & 3
            
            if j == 2:
                # Special: show both arithmetic and logical expectations
                match = "✓" if gpu_u == expected_logic else "✗"
                print(f"  {label:20s}: GPU={gpu_u:#018x}  logical_expect={expected_logic:#018x}  {match}")
            else:
                match = "✓" if gpu_u == expected else "✗"
                print(f"  {label:20s}: GPU={gpu_u:#018x}  expected={expected:#018x}  {match}")
    
    # Test negate_mod7
    print("\n" + "=" * 100)
    print("negate_mod7 test")
    print("=" * 100)
    
    # Test: negate of value 6 (binary 110) at bit position 4
    # p0=0, p1=bit4, p2=bit4
    inp = torch.tensor([0, 16, 16], dtype=torch.int64, device=device)
    outp = torch.zeros(3, dtype=torch.int64, device=device)
    
    test_negate_kernel[(1,)](inp, outp, num_warps=1)
    torch.cuda.synchronize()
    
    r = outp.cpu()
    print(f"Input:  p0={inp[0].item():#x}, p1={inp[1].item():#x}, p2={inp[2].item():#x}")
    print(f"Output: n0={r[0].item():#x}, n1={r[1].item():#x}, n2={r[2].item():#x}")
    
    # Expected: negate 6 (=110) -> 1 (=001)
    # is_zero = ~(0 | 16 | 16) = ~16 = 0xFFFF...EF
    # mask = ~is_zero = 16 (only bit 4 set)... WAIT
    # That's wrong! mask should have ALL bits set except where input is zero.
    # But `is_zero = ~(p0 | p1 | p2)` gives bits where ALL 3 planes are 0.
    # `mask = ~is_zero` gives bits where at least one plane is nonzero.
    # For our input: p0|p1|p2 = 0|16|16 = 16 (bit 4)
    # is_zero = ~16 = all bits EXCEPT bit 4
    # mask = 16 (ONLY bit 4)
    # n0 = ~0 & 16 = 0xFFFF... & 16 = 16
    # n1 = ~16 & 16 = (all except bit4) & 16 = 0
    # n2 = ~16 & 16 = 0
    # So result: n0=16, n1=0, n2=0 -> value 1 at bit 4. That's correct! (-6 mod 7 = 1)
    
    expected_n = [16, 0, 0]
    for k in range(3):
        rv = r[k].item()
        rv_u = rv if rv >= 0 else rv + (1<<64)
        match = "✓" if rv_u == expected_n[k] else "✗"
        print(f"  n{k}: GPU={rv_u:#018x}, expected={expected_n[k]:#018x} {match}")


if __name__ == "__main__":
    main()
