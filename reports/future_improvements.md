# Future Improvements

This document outlines practical next steps that could strengthen the project or potentially change the conclusions about Stage 2 value.

## 1. Larger Dataset Generation

### Current Limitation
- 5,000 rows total
- Only 117 borderline cases (2.3%)
- 45 borderline cases in test set

### Proposed Improvement
Generate a larger dataset:
- **Target size**: 50,000 - 100,000 rows
- **Expected borderline cases**: 1,000 - 2,000 (assuming similar distribution)
- **Benefit**: More training data for Stage 2 model, more statistical power for evaluation

### Implementation
```python
# In src/data_generation.py
df = create_dataset(n_rows=100000)
```

### Expected Impact
- More borderline cases for Stage 2 training
- More reliable borderline-subset metrics
- Better assessment of Stage 2 value

---

## 2. Richer Text Fields

### Current Limitation
Text fields are template-based with limited variation:
- `verification_note`: ~25 templates per class
- `ocr_document_text`: ~20 templates per class

### Proposed Improvement
Create more realistic text generation:
- **More templates**: 100+ per class
- **More variation**: Random typos, formatting inconsistencies
- **More context**: Longer, more detailed notes
- **Real-world patterns**: Study actual fraud investigation notes

### Implementation
Expand `data/external/` template files:
- `legit_verification_note_templates.csv`: 100+ templates
- `fraud_verification_note_templates.csv`: 100+ templates
- Add character-level noise injection

### Expected Impact
- Text features may capture patterns not in structured data
- More realistic evaluation of text model value

---

## 3. More Realistic Document Inconsistency Simulation

### Current Limitation
OCR text inconsistencies are simplistic:
- "Name partially unreadable"
- "Address mismatch"

### Proposed Improvement
Simulate realistic document fraud patterns:
- **Character substitution**: "Michael" → "Michae1" (1 for l)
- **Field transposition**: Wrong address on document
- **Partial matches**: First name matches, last name differs
- **Quality degradation**: Simulated low-quality scan artifacts

### Implementation
Add to `src/data_generation.py`:
```python
def generate_realistic_ocr_fraud(identity_fields):
    # Introduce realistic inconsistencies
    # - Character substitution
    # - Field swapping
    # - Partial information
    pass
```

### Expected Impact
- Text features may detect subtle inconsistencies
- More realistic fraud simulation

---

## 4. More Borderline Samples by Design

### Current Limitation
Borderline band (0.01-0.99) captures only 2.3% of cases because Stage 1 is very confident.

### Proposed Improvement
Generate more ambiguous cases intentionally:
- **Increase `legitimate_noisy` proportion**: 18% → 30%
- **Increase `true_name_fraud` proportion**: 4% → 10%
- **Add "hard" difficulty bias**: More medium/hard cases

### Implementation
Modify `src/data_generation.py` fraud mix:
```python
FRAUD_MIX = {
    'legitimate_clean': 0.55,      # Reduced from 0.70
    'legitimate_noisy': 0.25,      # Increased from 0.18
    'synthetic_identity': 0.05,
    'true_name_fraud': 0.10,       # Increased from 0.04
    'coordinated_attack': 0.05     # Increased from 0.03
}
```

### Expected Impact
- More borderline cases for Stage 2 training
- Better test of Stage 2 value on ambiguous cases

---

## 5. Richer Note-Risk Modeling

### Current Limitation
Current text features are simple:
- Keyword counts
- Text length
- Cosine similarity

### Proposed Improvement
More sophisticated text analysis:
- **Sentiment analysis**: Negative sentiment in notes
- **Named entity extraction**: Detect entity mismatches
- **Sequence patterns**: Order of information in notes
- **Fine-tuned classifier**: Train a small classifier on note risk

### Implementation
```python
# Add to src/encoder_features.py
def compute_note_sentiment(note):
    # Use a lightweight sentiment model
    pass

def extract_entities(text):
    # Use spaCy NER
    pass
```

### Expected Impact
- Richer text signal
- Potentially better Stage 2 performance

---

## 6. Better Calibrated Routing Threshold

### Current Limitation
Borderline band is fixed at [0.01, 0.99] based on visual inspection.

### Proposed Improvement
Optimize the routing threshold:
- **Grid search**: Test multiple threshold pairs
- **Validation-based selection**: Choose threshold that maximizes borderline-subset performance
- **Confidence calibration**: Use calibrated probabilities for routing

### Implementation
```python
def optimize_routing_threshold(val_df, thresholds):
    best_threshold = None
    best_metric = 0
    for low, high in thresholds:
        # Evaluate borderline-routed performance
        metric = evaluate_routed_system(val_df, low, high)
        if metric > best_metric:
            best_metric = metric
            best_threshold = (low, high)
    return best_threshold
```

### Expected Impact
- Better routing decisions
- Potentially improved Stage 2 contribution

---

## 7. More Segment-Level Validation

### Current Limitation
Validation focuses on overall and borderline subsets only.

### Proposed Improvement
Evaluate performance by:
- **Fraud type**: synthetic_identity, true_name_fraud, coordinated_attack
- **Difficulty level**: easy, medium, hard
- **Application characteristics**: thin_file, high_velocity, etc.

### Implementation
```python
def evaluate_by_segment(df, segment_col, setups):
    results = []
    for segment in df[segment_col].unique():
        segment_df = df[df[segment_col] == segment]
        for setup in setups:
            metrics = evaluate_setup(segment_df, setup)
            metrics['segment'] = segment
            results.append(metrics)
    return pd.DataFrame(results)
```

### Expected Impact
- Identify specific segments where Stage 2 helps
- More nuanced understanding of model behavior

---

## 8. Cost-Sensitive Evaluation

### Current Limitation
Current evaluation treats false positives and false negatives equally.

### Proposed Improvement
Incorporate business costs:
- **False negative cost**: Fraud loss (e.g., $5,000 average)
- **False positive cost**: Customer friction + review cost (e.g., $50)
- **Optimize for total cost**: Not just accuracy

### Implementation
```python
def compute_total_cost(y_true, y_pred, fn_cost=5000, fp_cost=50):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = fn * fn_cost + fp * fp_cost
    return total_cost
```

### Expected Impact
- More realistic evaluation
- May change optimal threshold and architecture

---

## 9. Alternative Encoder Models

### Current Limitation
Only tested MiniLM (all-MiniLM-L6-v2).

### Proposed Improvement
Test alternative encoders:
- **E5-small**: Potentially better for similarity tasks
- **BGE-small**: Good for retrieval/matching
- **Domain-specific**: Fine-tuned on financial text (if available)

### Implementation
```python
# In src/encoder_features.py
ENCODER_MODELS = [
    'sentence-transformers/all-MiniLM-L6-v2',
    'intfloat/e5-small-v2',
    'BAAI/bge-small-en-v1.5'
]
```

### Expected Impact
- Better text representations
- Potentially improved Stage 2 performance

---

## 10. Real Data Validation

### Current Limitation
All evaluation is on synthetic data.

### Proposed Improvement
If real data becomes available:
- **Validate patterns**: Do synthetic fraud patterns match real fraud?
- **Retrain models**: Train on real data
- **A/B test**: Compare Stage 1 vs Stage 2 in production

### Expected Impact
- Ground truth validation
- Real-world performance assessment

---

## Priority Ranking

| Priority | Improvement | Effort | Expected Impact |
|----------|-------------|--------|-----------------|
| 1 | Larger dataset generation | Low | High |
| 2 | More borderline samples by design | Low | High |
| 3 | Richer text fields | Medium | Medium |
| 4 | Segment-level validation | Low | Medium |
| 5 | Cost-sensitive evaluation | Low | Medium |
| 6 | Better routing threshold | Medium | Medium |
| 7 | Realistic OCR inconsistencies | Medium | Medium |
| 8 | Richer note-risk modeling | High | Medium |
| 9 | Alternative encoder models | Medium | Low |
| 10 | Real data validation | High | High |

---

## Conclusion

The most impactful improvements are:
1. **Generate more data** (especially borderline cases)
2. **Create more ambiguous cases by design**
3. **Enrich text field variation**

These changes could potentially shift the conclusion about Stage 2 value. However, it's also possible that even with these improvements, the structured model remains sufficient—which would further validate the current finding.
