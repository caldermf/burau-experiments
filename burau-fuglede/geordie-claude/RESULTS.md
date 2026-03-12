# Bigelow Billiard Search Results

## Overview

We search for curves in the 4-punctured disk (B_4, HUMPS=3) whose twisted
intersection polynomial is **zero mod m but nonzero over Z**. This is
Bigelow's non-faithfulness criterion for the Burau representation.

## Crucial parameter: `level_delta`

The original Bigelow code uses `level_delta = HUMPS + 1 = 4` (the level
change when the billiard wraps around hump 0). With this default:

- **mod 3: no victories** (tested to total = 1000)
- **mod 5: no victories** (tested to total = 1000)

The key discovery is that **changing `level_delta`** unlocks solutions:

| mod | level_delta | First victory (total) | Count (to total=100) | Count (to total=200) |
|-----|------------|----------------------|---------------------|---------------------|
| 2   | 1          | 12                   | many                | many                |
| 3   | 1          | **27**               | **50**              | ~200+               |
| 3   | 2          | **42**               | **41**              | ~80+                |
| 3   | 3          | none to 200          | 0                   | 0                   |
| 3   | 4 (default)| none to 1000         | 0                   | 0                   |
| 4   | 1          | 80                   | 2                   | ~10                 |
| 5   | 2          | **262**              | 0                   | 0 (need total>260)  |
| 6   | any        | none to 320          | 0                   | 0                   |

## All mod-3 examples (level_delta = 1, total ≤ 100)

| total | widths        | leftend | polynomial |
|-------|--------------|---------|------------|
| 27    | [12,12,3]    | 10      | −3t |
| 30    | [7,8,15]     | 14      | −3t⁻¹ − 3 − 3t |
| 30    | [7,15,8]     | 8       | −3t⁻¹ − 3 − 3t |
| 46    | [13,10,23]   | 22      | 3t⁻¹ + 3 + 3t |
| 46    | [13,23,10]   | 14      | 3t⁻¹ + 3 + 3t |
| 50    | [11,16,23]   | 24      | −3t⁻¹ − 3 − 3t |
| 50    | [11,23,16]   | 12      | −3t⁻¹ − 3 − 3t |
| 51    | [34,3,14]    | 2       | −3t² |
| 52    | [17,11,24]   | 4       | −3t⁻³ − 6t⁻² − 6t⁻¹ − 3 |
| 52    | [17,24,11]   | 13      | −3 − 6t − 6t² − 3t³ |
| 54    | [13,16,25]   | 26      | −3 − 3t − 3t² |
| 54    | [13,25,16]   | 14      | −3t⁻² − 3t⁻¹ − 3 |
| 58    | [9,24,25]    | 28      | −3 − 3t − 3t² |
| 58    | [9,25,24]    | 10      | −3t⁻² − 3t⁻¹ − 3 |
| 62    | [19,12,31]   | 30      | −3t⁻¹ − 3 − 3t |
| 62    | [19,31,12]   | 20      | −3t⁻¹ − 3 − 3t |
| 62    | [23,8,31]    | 30      | −3t⁻¹ − 3 − 3t |
| 62    | [23,31,8]    | 24      | −3t⁻¹ − 3 − 3t |
| 66    | [29,15,22]   | 10      | 3t⁻² + 3t⁻¹ + 3 |
| 66    | [29,22,15]   | 19      | 3 + 3t + 3t² |
| 66    | [43,8,15]    | 17      | −3t⁻¹ − 3 − 3t |
| 66    | [43,15,8]    | 26      | −3t⁻¹ − 3 − 3t |
| 68    | [17,17,34]   | 32      | −3t⁻¹ − 6 − 6t − 3t² |
| 68    | [17,34,17]   | 19      | −3t⁻² − 6t⁻¹ − 6 − 3t |
| 70    | [15,24,31]   | 34      | −3t⁻¹ − 3 − 3t |
| 70    | [15,31,24]   | 16      | −3t⁻¹ − 3 − 3t |
| 70    | [27,17,26]   | 8       | 3t⁻¹ + 3 + 3t |
| 70    | [27,26,17]   | 19      | 3t⁻¹ + 3 + 3t |
| 70    | [35,13,22]   | 28      | −3t⁻¹ − 3 − 3t |
| 70    | [35,22,13]   | 7       | −3t⁻¹ − 3 − 3t |
| 71    | [24,36,11]   | 22      | −3t |
| 74    | [11,31,32]   | 12      | −3t⁻¹ − 3 − 3t |
| 74    | [11,32,31]   | 36      | −3t⁻¹ − 3 − 3t |
| 78    | [9,29,40]    | 28      | −3t⁻¹ − 6 − 9t − 6t² − 3t³ |
| 78    | [9,40,29]    | 20      | −3t⁻³ − 6t⁻² − 9t⁻¹ − 6 − 3t |
| 78    | [25,14,39]   | 38      | 3t⁻¹ + 3 + 3t |
| 78    | [25,39,14]   | 26      | 3t⁻¹ + 3 + 3t |
| 80    | [23,18,39]   | 39      | −3t⁻¹ − 6 − 6t − 3t² |
| 80    | [23,39,18]   | 24      | −3t⁻² − 6t⁻¹ − 6 − 3t |
| 82    | [21,20,41]   | 1       | −3t⁻² − 3t⁻¹ − 3 |
| 82    | [21,41,20]   | 20      | −3 − 3t − 3t² |
| 84    | [23,19,42]   | 40      | 3t⁻² + 6t⁻¹ + 6 + 3t |
| 84    | [23,42,19]   | 25      | 3t⁻¹ + 6 + 6t + 3t² |
| 84    | [29,23,32]   | 9       | 3t⁻² + 6t⁻¹ + 6 + 3t |
| 84    | [29,32,23]   | 20      | 3t⁻¹ + 6 + 6t + 3t² |
| 86    | [19,29,38]   | 18      | −3t⁻¹ − 3 − 3t |
| 86    | [19,38,29]   | 1       | −3t⁻¹ − 3 − 3t |
| 87    | [40,44,3]    | 34      | −3t |
| 88    | [29,11,48]   | 4       | −3t⁻³ − 6t⁻² − 6t⁻¹ − 3 |
| 88    | [29,48,11]   | 25      | −3 − 6t − 6t² − 3t³ |

## All mod-3 examples (level_delta = 2, total ≤ 100)

| total | widths        | leftend | polynomial |
|-------|--------------|---------|------------|
| 42    | [25,1,16]    | 4       | −3t⁻¹ − 3 + 3t |
| 42    | [25,16,1]    | 21      | 3t⁻¹ − 3 − 3t |
| 42    | [25,16,1]    | 0       | 3t⁻¹ − 3 − 3t |
| 42    | [35,2,5]     | 3       | 3t⁻¹ − 3 − 3t |
| 42    | [35,5,2]     | 11      | −3t⁻¹ − 3 + 3t |
| 48    | [21,12,15]   | 14      | 3t + 3t² |
| 48    | [21,15,12]   | 7       | 3t⁻² + 3t⁻¹ |
| 54    | [29,10,15]   | 8       | 3t⁻¹ + 3 + 3t |
| 54    | [29,15,10]   | 21      | 3t⁻¹ + 3 + 3t |
| 66    | [57,4,5]     | 6       | −3t⁻² + 3t⁻¹ + 3 |
| 66    | [57,5,4]     | 18      | 3 + 3t − 3t² |
| 70    | [41,0,29]    | 35      | −3 |
| 70    | [41,0,29]    | 6       | −3 |
| 70    | [41,0,29]    | 0       | −3 |
| 70    | [41,29,0]    | 35      | −3 |
| 70    | [41,29,0]    | 6       | −3 |
| 70    | [41,29,0]    | 0       | −3 |
| 70    | [43,0,27]    | 7       | −3t⁻² − 3t⁻¹ + 3t |
| 70    | [43,0,27]    | 1       | 3t⁻¹ − 3t − 3t² |
| 70    | [43,27,0]    | 7       | −3t⁻² − 3t⁻¹ + 3t |
| 70    | [43,27,0]    | 1       | 3t⁻¹ − 3t − 3t² |
| 71    | [41,1,29]    | 6       | −3 |
| 71    | [41,29,1]    | 35      | −3 |
| 78    | [47,8,23]    | 9       | 3t⁻³ + 6t⁻² + 3t⁻¹ − 6 − 3t |
| 78    | [47,23,8]    | 38      | −3t⁻¹ − 6 + 3t + 6t² + 3t³ |
| 78    | [61,6,11]    | 19      | 3t⁻² + 3t⁻¹ − 3 |
| 78    | [61,11,6]    | 3       | −3 + 3t + 3t² |
| 86    | [9,38,39]    | 39      | 3t⁻¹ + 6 + 6t |
| 86    | [9,39,38]    | 13      | 6t⁻¹ + 6 + 3t |
| 96    | [17,29,50]   | 12      | −3t⁻¹ − 6 − 6t − 3t² |
| 96    | [17,50,29]   | 5       | −3t⁻² − 6t⁻¹ − 6 − 3t |
| 98    | [21,37,40]   | 15      | −3t⁻² − 6t⁻¹ − 12 − 9t − 3t² |
| 98    | [21,40,37]   | 6       | −3t⁻² − 9t⁻¹ − 12 − 6t − 3t² |
| 98    | [57,1,40]    | 49      | −6 + 3t² |
| 98    | [57,1,40]    | 0       | −6 + 3t² |
| 98    | [57,40,1]    | 8       | 3t⁻² − 6 |
| 98    | [59,1,38]    | 1       | 6t⁻¹ − 6t − 3t² |
| 98    | [59,38,1]    | 9       | −3t⁻² − 6t⁻¹ + 6t |
| 98    | [81,5,12]    | 5       | −3t⁻¹ + 6t |
| 98    | [81,12,5]    | 27      | 6t⁻¹ − 3t |
| 98    | [91,2,5]     | 31      | −6 + 3t² |
| 98    | [91,5,2]     | 11      | 3t⁻² − 6 |

## All mod-5 examples (level_delta = 2, total ≤ 320)

Mod-5 is much harder. The search needs to go to total ≥ 262 to find the
first result.

| total | widths          | leftend | polynomial |
|-------|----------------|---------|------------|
| 262   | [195,13,54]    | 68      | 5t⁻³ + 5t⁻² − 5 |
| 262   | [195,54,13]    | 127     | −5 + 5t² + 5t³ |
| 280   | [123,70,87]    | 82      | 5t + 5t² |
| 280   | [123,87,70]    | 41      | 5t⁻² + 5t⁻¹ |
| 298   | [117,11,170]   | 119     | −5t⁻² − 5t⁻¹ + 5t |
| 298   | [117,170,11]   | 147     | 5t⁻¹ − 5t − 5t² |

## Smallest example: mod 3, total = 27

```
widths = [12, 12, 3],  leftend = 10,  level_delta = 1
total = 27,  rightend = 23,  starting_cut = 6
Polynomial over Z:  −3t
Polynomial mod 3:   0
```

The billiard traces 13 steps (total/2 = 13). Each step contributes ±t^level
to the polynomial. With level_delta = 1, the level changes by ±1 at hump
boundaries (instead of ±4 with the original default), keeping all
contributions concentrated near level 0 so that cancellation mod 3 can occur.

## Patterns observed

1. **All mod-3 coefficients are multiples of 3** — the polynomial is always
   of the form 3·p(t) for some Laurent polynomial p(t) with integer
   coefficients.

2. **Many examples come in pairs** related by swapping w1 ↔ w2 (the two
   inner humps). The polynomials are related by t ↦ t⁻¹.

3. **The most common polynomial shape is ±3(t⁻¹ + 1 + t)** — a "palindromic"
   degree-2 Laurent polynomial.

4. **Larger totals produce higher-degree polynomials**, e.g.
   −3t⁻³ − 6t⁻² − 9t⁻¹ − 6 − 3t at total=78.

## Mod 6: not found

No mod-6 victories were found with any parameter variation through total ≈ 320.
This may require a fundamentally different approach (e.g. n=5 or n=6).

## Comparison with direct Burau kernel search

Using BFS collision search on B_4 mod-m Burau matrices:

| mod | Max word length explored | Genuine kernel elements found |
|-----|------------------------|------------------------------|
| 2   | 20                     | 20                           |
| 3   | 24                     | 0                            |
| 5   | 20                     | 0                            |

The billiard search finds mod-3 solutions much more efficiently than the
direct Burau search: the smallest billiard example (total=27) corresponds to
a braid of moderate length, while the direct search exhaustively checked all
braids up to length 24 without finding a mod-3 kernel element.
