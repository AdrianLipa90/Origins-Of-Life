# v0.10 Non-arbitrary Pass Without Probability Closure

This update corrects v0.9. The model is allowed to pass its derived proxy threshold, but it is not allowed to claim factual closure.

## Core Result

- proxy_pass: `True`
- fact_claimed: `False`
- score: `0.708705`
- threshold: `0.692709`
- threshold_margin: `0.015996`
- truth_proximity_probability: `0.708705`
- nonclosure_margin_to_one: `0.291295`

## Non-arbitrary Rule

The threshold is re-derived from v0.5:

```text
threshold = grammar_compression_proxy * complement_selectivity_proxy
```

No manual score adjustment, no manual threshold adjustment, no new tuning constants, and no probability closure to 1 are used.

## Meaning

The model passes as a protocell-candidate proxy. It does not claim empirical abiogenesis as a fact. The ontological aperture remains open as a nonzero margin to 1.
