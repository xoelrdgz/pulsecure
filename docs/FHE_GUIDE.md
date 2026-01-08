# Fully Homomorphic Encryption (FHE) Guide

## What is FHE?

Fully Homomorphic Encryption allows computations on encrypted data without decrypting it. The result, when decrypted, matches what you would get from computing on plaintext.

```
Encrypt(A) + Encrypt(B) = Encrypt(A + B)
Encrypt(A) * Encrypt(B) = Encrypt(A * B)
```

A server can process data without ever seeing it.

## Why FHE for Medical Data?

| Traditional ML | FHE-ML |
|----------------|--------|
| Server sees patient data | Server sees only ciphertext |
| Trust the server | Trust the math |
| Breach = data leaked | Breach = encrypted garbage |

## The Noise Problem

FHE ciphertexts contain noise that grows with each operation:

```
Initial noise:     ####............  (low)
After additions:   ########........  (medium)  
After multiplies:  ################  (high, decryption fails)
```

### Noise Budget

- Each ciphertext has a noise budget
- Additions consume little budget
- Multiplications consume more budget
- When budget exhausted, decryption fails

### Pulsecure Strategy

1. Logistic Regression: Linear model with few multiplications
2. 12-bit Quantization: Fewer bits means less noise per operation
3. Lookup Tables: Sigmoid computed via LUT (programmable bootstrapping)

## FHE Schemes

| Scheme | Strengths | Used By |
|--------|-----------|---------|
| BGV | Efficient integers | HElib |
| BFV | Similar to BGV | SEAL |
| CKKS | Approximate floats | OpenFHE |
| TFHE | Fast bootstrapping | tfhe-rs |

### Why tfhe-rs?

- Rust-native: No FFI overhead, memory safety
- Active development by Zama
- Programmable bootstrapping for lookup tables
- Fast noise refresh

## Key Types

```rust
ClientKey {
    secret_key: [u8],  // Decryption capability, NEVER share
}

ServerKey {
    bootstrapping_key: [...],  // Noise refresh
    key_switching_key: [...],  // Operation support
}
```

### Security Model

- ClientKey: Keep secret, enables decryption
- ServerKey: Give to compute server, cannot decrypt
- Separation: Mathematically guaranteed privacy

## Pulsecure FHE Pipeline

### Quantized Inference

The Python pipeline exports a quantized model:

```
1. Input: Patient features (10 values)
2. Normalize: Subtract mean, multiply by 1/std (quantized)
3. Linear: Dot product with coefficients (integer arithmetic)
4. Sigmoid: Lookup table (256 entries)
5. Calibration: Isotonic LUT (64 entries)
6. Threshold: Compare to clinical threshold
```

### Quantization Details

| Parameter | Value |
|-----------|-------|
| Precision | 12 bits |
| Scale Factor | 4096 |
| Accumulator | 64-bit integers |
| Sigmoid LUT | 256 entries |
| Calibration LUT | 64 entries |

The 12-bit quantization preserves AUC within 0.001 of the float model.

## Performance

| Operation | Latency |
|-----------|---------|
| Key Generation | 2-5s |
| Encrypt (per value) | 10ms |
| FHE Add | 1ms |
| FHE Multiply | 50ms |
| Bootstrap | 100ms |
| Full Inference | 1-5s |

FHE is 1,000-10,000x slower than plaintext computation.

## Security Guarantees

- 128-bit security (equivalent to AES-128)
- Post-quantum secure (many schemes resist quantum attacks)
- Semantic security (ciphertexts reveal nothing about plaintext)

## Further Reading

- [tfhe-rs Documentation](https://docs.zama.ai/tfhe-rs)
- [FHE.org Resources](https://fhe.org)
- [Homomorphic Encryption Standard](https://homomorphicencryption.org)
