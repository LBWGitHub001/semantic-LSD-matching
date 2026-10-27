# SemanticLineMatcher

Independent dual-output matcher (does not replace `XFeatLineMatcher`).

## Outputs

- `log_P` / `P`: region bipartite matching with dustbin
- `log_P_line` / `P_line`: line bipartite matching with dustbin
- `line_parents0/1`: parent region index per flattened line

## Line descriptor

1. Shared CNN on each region patch → feature map
2. Region vector `f_r` via global pool
3. Along-line sample + 1D aggregate → `f_line`
4. `d = Proj([f_line || f_r])` for line Sinkhorn

## Loss (`SemanticLineMatchingLoss`)

`L = λ_R L_region_KL + λ_L L_line_KL + λ_S L_sem_JSD + λ_G L_geo_adjacency`

- Region / line KL: SuperGlue-style assignment NLL (hard-label KL)
- Semantic concentration: generalized JSD over line→semantic marginals within each parent region (A→B and B→A)
- Geo: adjacency preservation — soft `A0 ≈ P A1 P^T` ("neighbors stay neighbors across views")

Registry name: `semantic_line_matcher`.
