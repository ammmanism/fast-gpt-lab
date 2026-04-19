# Mathematical Foundations — BPE Tokenizer

## 1. Problem Statement

Natural language vocabulary is open-ended. Naïve word-level tokenizers fail on:
- Out-of-vocabulary words (OOV)
- Morphological variations ("running", "runs", "ran")
- Multilingual corpora

Character-level tokenizers solve OOV but produce very long sequences (slow attention).

**BPE** (Byte-Pair Encoding) finds the **optimal compression** between these extremes.

---

## 2. BPE Algorithm (Formal)

**Input**: Corpus $\mathcal{C}$ as sequence of bytes, target vocab size $V$.

**Initialization**: $\mathcal{V}_0 = \{0, 1, \ldots, 255\}$ (byte alphabet)

**Iteration** for $k = 1, \ldots, V - 256$:
1. Count all adjacent pair frequencies:
   $$f(a, b) = \sum_{w \in \mathcal{C}} \text{count}(a, b \text{ adjacent in } w)$$
2. Select best merge: $(a^*, b^*) = \arg\max_{(a,b)} f(a, b)$
3. Add new token: $\mathcal{V}_k = \mathcal{V}_{k-1} \cup \{a^*b^*\}$
4. Replace all $(a^*, b^*)$ adjacencies in corpus with $a^*b^*$

**Output**: Vocabulary $\mathcal{V}_V$ and merge table $\mathcal{M} = \{(a_k, b_k)\}_{k=1}^{V-256}$

---

## 3. Complexity Analysis

| Phase | Time | Space |
|-------|------|-------|
| Count pairs | $O(N)$ per iteration | $O(\|\mathcal{V}\|^2)$ |
| Apply merge | $O(N)$ worst case | $O(N)$ |
| Total (BPE training) | $O(N \cdot V)$ | $O(N + V^2)$ |

For GPT-2: $N \approx 10^{10}$ tokens, $V = 50257$.

In practice, the inner loop is dominated by Python dict operations — which is why production tokenizers (tiktoken, HuggingFace Fast) are implemented in Rust/C++.

---

## 4. Encoding Complexity

Given trained merges, encoding a string of length $N$:
- Naïve: $O(N \cdot \|\mathcal{M}\|)$ — applies each merge greedily
- Optimal: $O(N \log N)$ via priority queue on pair frequencies

Our implementation uses the naïve approach with a word-level cache (`self._cache`) which makes it $O(W \cdot \|\mathcal{M}\|)$ amortised where $W$ = unique words.

---

## 5. Byte-Level BPE (GPT-2 Innovation)

Standard BPE operates on characters → fails on non-UTF8 bytes.

GPT-2's innovation: map every byte $\in [0, 255]$ to a unique printable unicode character, then run BPE on characters. This guarantees:

$$\forall \text{ string } s: \text{encode}(\text{decode}(\text{encode}(s))) = \text{encode}(s)$$

i.e., the encode→decode→encode roundtrip is **idempotent**. Our `_build_byte_encoder` implements this bijection.

---

## 6. Compression Ratio

For English text with GPT-2 tokenizer:

$$\rho = \frac{\text{tokens}}{\text{bytes}} \approx 0.25$$

i.e., 4 bytes compress to ~1 token on average.
For code: $\rho \approx 0.35$ (less compressive — more unique symbols).
For Chinese: $\rho \approx 0.5$ (each character is 3 bytes UTF-8 but ~1 token).
