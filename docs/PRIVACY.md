# Differential Privacy Guide

## What is Differential Privacy?

Differential Privacy (DP) provides a mathematical guarantee that the output of a computation does not reveal whether any individual's data was included.

## The Core Idea

```
Query: "What percentage of patients have heart disease?"

Without DP:  "42.3%"     -> Might identify individuals
With DP:     "41-44%"    -> Protects individuals, still useful
```

By adding calibrated noise, we protect individuals while preserving aggregate utility.

## Epsilon: The Privacy Budget

Epsilon quantifies privacy loss:

| Epsilon | Privacy Level | Noise Added |
|---------|---------------|-------------|
| 0.1 | Very high privacy | High noise |
| 1.0 | Standard privacy | Moderate noise |
| 10.0 | Low privacy | Low noise |

Lower epsilon means more privacy, more noise, and less accuracy.

## Mechanisms

### Laplacian Mechanism

For numeric queries, add noise from Laplace distribution:

```
true_answer + Laplace(0, sensitivity/epsilon)
```

Where sensitivity is the maximum change from one person's data.

### Gaussian Mechanism

Alternative using normal distribution (requires epsilon-delta DP):

```
true_answer + Normal(0, sigma)
```

## Privacy Budget Composition

Each query consumes privacy budget:

```
Query 1: epsilon = 0.5
Query 2: epsilon = 0.3
Query 3: epsilon = 0.2
Total:   epsilon = 1.0
```

Once budget is exhausted, no more queries can be answered privately.

## Application in Pulsecure

### What DP Protects

1. Aggregate statistics: "X% of predictions were positive"
2. Model performance metrics: "Accuracy was approximately Y%"
3. Usage patterns: "Z predictions made this week"

### What DP Does Not Protect

- Individual predictions (protected by FHE instead)
- Raw patient data (never leaves client without encryption)

DP protects aggregated outputs that could leak through statistical inference.

## Re-identification Attacks

Without DP, aggregates can leak individual data:

```
"Average age of heart disease patients: 52.3"
"Average age after removing Alice: 52.1"
-> Alice's age can be computed from difference
```

DP noise prevents this attack:

```
"Average age (DP): 51-54"
"Average after removing Alice (DP): 50-55"
-> No useful information leaked
```

## Configuration

```rust
pub struct PrivacyConfig {
    pub max_epsilon: f64,        // Maximum budget per session
    pub epsilon_per_query: f64,  // Budget per query
    pub use_laplace: bool,       // Laplace vs Gaussian
}
```

## Further Reading

- [Algorithmic Foundations of DP](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
- [OpenDP Documentation](https://docs.opendp.org)
- [Programming Differential Privacy](https://programming-dp.com)
