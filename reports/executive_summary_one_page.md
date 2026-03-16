# Executive Summary: Identity Fraud Detection Project

## One-Page Overview

---

### Project

**Selective Encoder-Enhanced Identity Fraud Detection for Application Risk**

### Objective

Test whether lightweight text/encoder features improve fraud detection on borderline cases where a structured model is uncertain.

---

### System Architecture

```
Application → Stage 1 (Structured) → Confident? → Yes → Use Stage 1 (97.7%)
                                          ↓
                                         No → Stage 2 (Text) → Combined (2.3%)
```

---

### Key Results

| Metric | Stage 1 Only | With Text (All) | Borderline Routed |
|--------|--------------|-----------------|-------------------|
| **ROC-AUC** | **0.995** | 0.957 | 0.993 |
| Precision | 96.7% | 96.7% | 96.7% |
| Recall | 89.0% | 89.0% | 89.0% |

**Bottom Line**: Adding text features provides **no improvement** and can hurt performance.

---

### Why Stage 2 Didn't Help

1. **Strong baseline**: 99.5% ROC-AUC leaves almost no room for improvement
2. **Feature correlation**: Text features capture similar patterns as structured features
3. **Small borderline set**: Only 117 cases limited Stage 2 training and evaluation

---

### Recommendation

**Use Stage 1 (LightGBM) alone.**

The two-stage architecture adds complexity (encoder dependency, extra latency, more models to maintain) without improving accuracy.

---

### Key Findings

| Finding | Implication |
|---------|-------------|
| Structured features are sufficient | No need for NLP complexity |
| Text on all cases hurts | Feature dilution effect |
| Borderline routing preserves performance | Safest architecture if text must be used |
| Strong baselines make improvement hard | Establish baseline before adding complexity |

---

### Project Value

This project demonstrates:
- ✅ Rigorous experimental methodology (ablation study)
- ✅ Honest evaluation of negative results
- ✅ Production-oriented thinking (latency, complexity)
- ✅ Understanding that simpler can be better

**Transferable insight**: Always validate whether added complexity provides value.

---

### Technical Stack

| Component | Choice |
|-----------|--------|
| Stage 1 Model | LightGBM |
| Stage 2 Encoder | MiniLM (sentence-transformers) |
| Stage 2 Combiner | Logistic Regression |
| Dataset | 5,000 synthetic applications |
| Fraud Rate | 12% |

---

### Contact

Built as a portfolio project demonstrating end-to-end ML system design, rigorous experimentation, and honest evaluation of results.
