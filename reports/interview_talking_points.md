# Interview Talking Points

## Quick Pitch (30 seconds)

> "I built a two-stage fraud detection system to test whether text analysis could improve predictions on uncertain cases. The structured model achieved 99.5% ROC-AUC, and my rigorous ablation study showed that text features provided no incremental value. This taught me that strong feature engineering can eliminate the need for complex NLP, and that negative results are valuable when properly validated."

---

## Project Overview (2 minutes)

### The Problem
"Application fraud is a significant problem in financial services. Fraudsters submit fake or stolen identities to open accounts. The challenge is detecting fraud while minimizing friction for legitimate customers."

### My Approach
"I designed a two-stage system:
- Stage 1 uses structured features like device velocity and identity reuse patterns
- Stage 2 adds text analysis for borderline cases where Stage 1 is uncertain
- The hypothesis was that text features would catch fraud that structured features miss"

### Key Results
"The structured model achieved 99.5% ROC-AUC with 97% precision and 89% recall. When I added text features, performance actually decreased slightly. My ablation study showed that the structured features already captured the fraud signal, making text features redundant."

### The Takeaway
"Not every problem needs deep learning or NLP. Strong feature engineering can be sufficient, and it's important to validate whether added complexity actually helps."

---

## Technical Deep Dives

### Feature Engineering

**Q: What features did you engineer?**

"I created 23 structured features in categories:
- **Velocity features**: device reuse in 7 days, email/phone/address reuse in 30 days
- **Tenure features**: months at address, months at employer, minimum of both
- **Risk flags**: thin file, night application, high identity reuse
- **Derived features**: income/age ratio, name-email match score

The most discriminative were:
- `high_identity_reuse_flag`: 67% fraud rate when true vs 1.4% when false
- `high_device_velocity_flag`: Only appeared in fraud cases
- `night_application_flag`: 7x higher in fraud"

### Model Selection

**Q: Why LightGBM over other models?**

"I compared Logistic Regression, LightGBM, and XGBoost:
- Logistic Regression: 0.988 ROC-AUC (good baseline)
- LightGBM: 0.995 ROC-AUC (best)
- XGBoost: 0.992 ROC-AUC (close second)

LightGBM won because it handles categorical features well, is fast to train, and achieved the best validation metrics. The 0.7% improvement over Logistic Regression is meaningful at scale."

### Text Features

**Q: What text features did you create?**

"I used a lightweight encoder (MiniLM) to create:
- **OCR similarity**: Cosine similarity between claimed identity and document text
- **Consistency scores**: Similarity between employer/address info and explanations
- **Keyword features**: Count of suspicious words like 'mismatch', 'unverifiable'
- **Length features**: Text lengths as anomaly indicators

The text-only model achieved 82% ROC-AUC on borderline cases, showing text has signal—but the structured model already captured it."

### Ablation Study

**Q: How did you validate the two-stage approach?**

"I ran a rigorous ablation study with four setups:
1. Structured only (baseline)
2. Text only (on borderline cases)
3. Combined on all cases
4. Borderline routing (combined only for uncertain cases)

Results showed:
- Structured only: 0.995 ROC-AUC (best)
- Combined all: 0.957 ROC-AUC (worse!)
- Borderline routed: 0.993 ROC-AUC (preserves performance)

The key insight: adding text to all cases hurts because it dilutes the strong structured signal."

---

## Handling Tough Questions

### "Why didn't the text features help?"

"Three reasons:
1. **Strong baseline**: With 99.5% ROC-AUC, there's almost no room for improvement
2. **Feature correlation**: Text features captured similar patterns as structured features
3. **Small borderline set**: Only 117 borderline cases limited Stage 2 training

This doesn't mean text never helps—it means for this dataset, structured features were sufficient."

### "Isn't this a failed project?"

"No, it's a successful validation of a hypothesis. The scientific method includes testing ideas that don't pan out. I learned:
- Strong feature engineering can eliminate the need for complex models
- Ablation studies are essential before adding complexity
- Negative results are valuable when properly documented

In production, this saves the cost of maintaining an encoder model that provides no benefit."

### "What would you do differently?"

"Three things:
1. **Generate more borderline cases**: The 2.3% borderline rate limited Stage 2 learning
2. **Richer text simulation**: More realistic OCR errors and note variation
3. **Test alternative encoders**: Only tried MiniLM; others might perform better

But I'd also accept that the structured model might just be sufficient for this problem."

### "How would this work in production?"

"For this dataset, I'd deploy Stage 1 only—simpler and equally effective.

If we wanted to keep the two-stage option:
- Stage 1 runs on all applications (~40ms)
- Only 2.3% of cases route to Stage 2 (~50ms encoder + inference)
- Average latency increase: ~1ms per application

The infrastructure cost of maintaining the encoder isn't justified by the performance gain (none)."

---

## Behavioral Questions

### "Tell me about a time you had to deliver bad news."

"In this project, I had to conclude that the text features I spent significant time building provided no value. Rather than hiding this or cherry-picking metrics, I documented it honestly in the validation report. I framed it as a successful experiment—we tested a hypothesis and got a clear answer. My manager/stakeholder would rather know early that a feature doesn't help than deploy complexity that provides no benefit."

### "Tell me about a time you simplified a complex problem."

"The original design had a complex two-stage system with encoder models and routing logic. After validation showed no benefit, I recommended using the simple structured model alone. This reduced:
- Model complexity (1 model instead of 3)
- Runtime (no encoder overhead)
- Maintenance burden (no NLP dependencies)

The simpler solution was actually better."

### "How do you handle ambiguity?"

"In this project, the borderline cases were inherently ambiguous—that's why we routed them to Stage 2. I handled this by:
1. Defining clear criteria (score between 0.01 and 0.99)
2. Measuring performance specifically on the ambiguous subset
3. Accepting that some ambiguity is irreducible

The validation showed that even with text features, borderline cases remained hard to classify—which is expected for truly ambiguous cases."

---

## Questions to Ask the Interviewer

1. "How does your team handle negative experimental results? Do you have a culture of publishing/sharing what didn't work?"

2. "What's the balance between simple interpretable models and complex ML in your fraud detection stack?"

3. "How do you decide when to add complexity vs. improve feature engineering?"

4. "What's your approach to validating model improvements before production deployment?"

---

## Key Numbers to Remember

| Metric | Value |
|--------|-------|
| Dataset size | 5,000 applications |
| Fraud rate | 12% |
| Stage 1 ROC-AUC | 0.995 |
| Stage 1 Precision | 96.7% |
| Stage 1 Recall | 89.0% |
| Borderline cases | 2.3% (117 total) |
| Text-only ROC-AUC | 0.818 |
| Combined-all ROC-AUC | 0.957 (worse than Stage 1) |
| Routed ROC-AUC | 0.993 |
| Encoder latency | ~50ms per case |
