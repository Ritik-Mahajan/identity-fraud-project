# Final Presentation Outline

## 8-Slide Presentation Structure (10-15 minutes)

---

## Slide 1: Problem and Motivation

**Title**: Application Fraud: A Billion-Dollar Problem

**Purpose**: Set the business context and why this matters

**Bullets**:
- Application fraud costs financial institutions billions annually
- Fraudsters submit fake/stolen identities to open accounts
- Challenge: High precision (don't block legitimate customers) + High recall (catch fraud)
- Borderline cases require expensive manual review—automating these has high ROI
- Goal: Test whether text analysis can improve decisions on uncertain cases

**Suggested Visual**: 
- Simple diagram showing fraud impact (cost, volume, manual review burden)
- Or: Example of a borderline case (mixed signals)

**Speaker Notes**: "This project addresses a real problem in financial services. The key insight is that borderline cases—where signals are mixed—are both the hardest to decide and the most expensive to review manually."

---

## Slide 2: Two-Stage System Design

**Title**: Architecture: Selective Text Analysis

**Purpose**: Explain the system design and hypothesis

**Bullets**:
- **Stage 1**: Structured model (device velocity, identity reuse, tenure patterns)
- **Stage 2**: Text analysis (OCR similarity, keyword detection) for borderline cases only
- Hypothesis: Text features will catch fraud that structured features miss
- Borderline routing: Apply expensive text analysis only where needed (2.3% of cases)
- Production benefit: Efficient use of compute resources

**Suggested Visual**:
```
Application → Stage 1 → Confident? → Yes → Decision
                            ↓
                           No (2.3%)
                            ↓
                       Stage 2 → Decision
```

**Speaker Notes**: "The two-stage design is production-oriented. We don't want to run an encoder on every application—only on the uncertain ones where it might help."

---

## Slide 3: Synthetic Dataset Design

**Title**: Data: Realistic Fraud Simulation

**Purpose**: Explain dataset and why synthetic data is valid

**Bullets**:
- 5,000 applications, 12% fraud rate, 12-month time span
- Five fraud archetypes: Legitimate clean/noisy, Synthetic identity, True-name fraud, Coordinated attack
- Time-based train/val/test split (prevents leakage)
- Why synthetic: Real fraud data is sensitive, imbalanced, proprietary
- Patterns simulated are based on documented fraud behaviors

**Suggested Visual**:
| Type | % | Description |
|------|---|-------------|
| Legitimate Clean | 70% | Normal applications |
| Legitimate Noisy | 18% | Real customers, messy data |
| Synthetic Identity | 5% | Fabricated identities |
| True-Name Fraud | 4% | Real identity, fraud intent |
| Coordinated Attack | 3% | Fraud rings |

**Speaker Notes**: "Synthetic data lets me control fraud patterns and share the project. The patterns I simulated—device reuse, thin files, velocity—are well-documented in real fraud."

---

## Slide 4: Stage 1 Results

**Title**: Stage 1: Strong Structured Baseline

**Purpose**: Show that Stage 1 performs extremely well

**Bullets**:
- 23 engineered features across velocity, tenure, and risk categories
- LightGBM selected over Logistic Regression and XGBoost
- **Test ROC-AUC: 0.995** (near-perfect discrimination)
- **Precision: 96.7%** (low false positives)
- **Recall: 89.0%** (catches most fraud)

**Suggested Visual**:
| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| **LightGBM** | **0.995** | **0.974** |
| XGBoost | 0.992 | 0.960 |
| Logistic Regression | 0.988 | 0.944 |

**Speaker Notes**: "The structured model is extremely strong. This is important context for what comes next—with 99.5% ROC-AUC, there's very little room for improvement."

---

## Slide 5: Stage 2 Text/Encoder Experiment

**Title**: Stage 2: Testing the Text Hypothesis

**Purpose**: Explain what Stage 2 tried to do

**Bullets**:
- Used MiniLM encoder (lightweight, fast, local)
- 12 text features: OCR similarity, consistency scores, keyword counts, text lengths
- Applied to 117 borderline cases (scores between 0.01 and 0.99)
- Combined with Stage 1 score using Logistic Regression
- Hypothesis: Text will catch fraud that structured features miss

**Suggested Visual**:
| Feature | Description |
|---------|-------------|
| OCR similarity | Claimed identity vs document text |
| Employment consistency | Employer info vs explanation |
| Keyword count | Suspicious words detected |
| Text length | Anomaly indicator |

**Speaker Notes**: "Stage 2 used a lightweight encoder to avoid production latency issues. The goal was to test whether any text signal helps, not to build the most sophisticated NLP system."

---

## Slide 6: Ablation and Validation Findings

**Title**: Results: Text Features Add No Value

**Purpose**: Present the key finding with evidence

**Bullets**:
- **Ablation study**: Four configurations tested
- **Structured only**: 0.995 ROC-AUC (best)
- **Combined all cases**: 0.957 ROC-AUC (**worse!**)
- **Borderline routed**: 0.993 ROC-AUC (preserves, doesn't improve)
- **Borderline-specific**: 0% error reduction from text features

**Suggested Visual**:
| Setup | ROC-AUC | Verdict |
|-------|---------|---------|
| Structured Only | **0.995** | **Best** |
| Text Only | 0.818 | Weak |
| Combined (All) | 0.957 | Hurts |
| Borderline Routed | 0.993 | Preserves |

**Speaker Notes**: "This is the key finding. Adding text to all cases actually hurts performance. Even on borderline cases specifically, there's no improvement. The structured features already capture the fraud signal."

---

## Slide 7: Conclusion and Recommendation

**Title**: Recommendation: Stage 1 Only

**Purpose**: Translate findings into actionable recommendation

**Bullets**:
- **For this dataset**: Use Stage 1 (LightGBM) alone
- Two-stage architecture adds complexity without improving accuracy
- Saves: encoder maintenance, latency overhead, model dependencies
- **Key insight**: Strong feature engineering can eliminate need for complex NLP
- **Methodology transfers**: Baseline → Ablation → Validation applies to any "should we add X?" question

**Suggested Visual**:
| Factor | Stage 1 Only | Two-Stage |
|--------|--------------|-----------|
| ROC-AUC | 0.995 | 0.993 |
| Latency | ~40ms | ~90ms (borderline) |
| Complexity | 1 model | 3 models + encoder |
| Maintenance | Low | Higher |

**Speaker Notes**: "The recommendation is clear: use the simpler system. This is a valuable finding—it prevents deploying unnecessary complexity."

---

## Slide 8: Lessons Learned and Future Work

**Title**: What I Learned

**Purpose**: Demonstrate mature ML thinking and growth mindset

**Bullets**:
- **Lesson 1**: Strong baselines make improvement hard—establish baseline first
- **Lesson 2**: Ablation studies are essential before adding complexity
- **Lesson 3**: Negative results are valuable when properly validated
- **Future work**: Larger dataset, richer text simulation, alternative encoders
- **Transferable skill**: Rigorous validation methodology applies to any ML problem

**Suggested Visual**: None needed—let the bullets speak

**Speaker Notes**: "The biggest takeaway is that not every problem needs deep learning or NLP. Sometimes good feature engineering is enough. Knowing when NOT to add complexity is as important as knowing when to add it."

---

## Appendix Slides (If Time Permits)

### A1: Feature Engineering Details

**Bullets**:
- Velocity: device_reuse_7d, email_reuse_30d, phone_reuse_30d, address_reuse_30d
- Tenure: months_at_address, months_at_employer, tenure_min
- Risk flags: thin_file_flag, night_application_flag, high_identity_reuse_flag
- Derived: income_age_ratio, name_email_match_score

### A2: Calibration and Stability

**Bullets**:
- Brier score: 0.015 (well-calibrated)
- Stable across test months (no drift)
- Threshold analysis: Performance holds at 0.3-0.7 thresholds

### A3: Why Synthetic Data Is Valid

**Bullets**:
- Real fraud data is sensitive, imbalanced, proprietary
- Patterns simulated are based on documented fraud behaviors
- Methodology transfers regardless of data source
- Portfolio project constraint—not a production deployment

---

## Presentation Tips

1. **Practice the 30-second summary** before diving into slides
2. **Know your numbers**: 0.995 ROC-AUC, 96.7% precision, 89% recall, 2.3% borderline
3. **Anticipate questions**: "Why didn't text help?" "Is this a failed project?"
4. **End with the insight**: Strong feature engineering can be enough; validate before adding complexity
5. **Time check**: Aim for 10-12 minutes to leave time for questions
