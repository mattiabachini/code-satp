# Location Extraction Models: Experimental Plan
**Project**: SATP Indian Incident Location Extraction
**Date**: November 2025
**Goal**: Extract structured location data (state, district, village, other_locations) from incident summaries

---

## RESEARCH QUESTION
What's the best approach for extracting structured location information from Indian incident reports, 
and what is the privacy-performance trade-off between self-hosted vs. proprietary models?

---

## THREE-STAGE EXPERIMENTAL DESIGN

### STAGE 1: Architecture Screening (Base Models)
**Goal**: Identify which architecture and pre-training approach works best
**Timeline**: 1-2 weeks
**Cost**: $0 (Colab Free)

### STAGE 2: Scaling Study (Large Models)
**Goal**: Determine open-source performance ceiling
**Timeline**: 3-5 days  
**Cost**: $30-100
**Approach**: Scale only the top 2-3 performers from Stage 1

### STAGE 3: Modern Decoders (APIs & Large LLMs)
**Goal**: Compare best open-source vs. proprietary models
**Timeline**: 1-2 days
**Cost**: $1-20
**Approach**: Zero-shot evaluation of large decoders

---

## CANDIDATE MODELS

### BASELINES (Always test first)
1. **Dictionary/Gazetteer Lookup** - Pattern matching against Indian place name database
2. **SpaCy NER (en_core_web_sm)** - Pre-trained NER without fine-tuning
3. **Regex Patterns** - Rule-based extraction

**Expected**: 15-25% exact match
**Purpose**: Show simple methods are insufficient


### Favorites

**1. Seq2seq Models**
*Flan-T5-base*
*Flan-T5-Large*
*mt5-base*
*mt0-base*
*IndicBART*

**2. BERT Models**
*confliBERT*
*MuRIL-base*
*IndicBERT*
*XLM-RoBERTa-base*

**3. Modern Decoders**
*Gemini 1.5 Flash*
*GPT-4o-mini*
---

### STAGE 1: BASE MODELS (Fine-tune all on Colab Free)

#### A. Seq2seq Models (Encoder-Decoder)
| Model | Hugging Face ID | Size | Why Test |
|-------|-----------------|------|----------|
| T5-base | `t5-base` | 220M | Standard baseline |
| FLAN-T5-base | `google/flan-t5-base` | 250M | Instruction-tuned T5 |
| mT5-base | `google/mt5-base` | 580M | Multilingual T5 |
| mT0-base | `bigscience/mt0-base` | 580M | Multilingual + instruction-tuned |
| IndicBART | `ai4bharat/IndicBART` | 240M | Indian language specialist |

**Expected**: 40-50% exact match

**Note**: No conflict-specific seq2seq model exists (confliBERT is BERT-only)

#### B. BERT Models (Token Classification)
| Model | Hugging Face ID | Size | Why Test |
|-------|-----------------|------|----------|
| BERT-base | `bert-base-cased` | 110M | Standard BERT baseline |
| **confliBERT** | `snowood1/ConfliBERT-scr-uncased` | 110M | **Pre-trained on political conflict texts** |  
| MuRIL-base | `google/muril-base-cased` | 237M | Google's Indian BERT |
| IndicBERT | `ai4bharat/indic-bert` | 110M | AI4Bharat's Indian BERT |
| XLM-RoBERTa-base | `xlm-roberta-base` | 270M | Multilingual RoBERTa |

**Expected**: 45-50% exact match
**Note**: Requires different approach (token classification vs. seq2seq generation)

**confliBERT Details**:
- Pre-trained on large corpus of political conflict and violence texts
- Should excel at understanding conflict incident language
- Likely better at recognizing conflict-related entities and context
- Important comparison: domain-specific (conflict) vs. language-specific (Indian)
---

### STAGE 2: LARGE MODELS (Fine-tune top 2-3 from Stage 1)

#### Fine-Tuning Options
| Model | Size | Platform | Cost | Expected |
|-------|------|----------|------|----------|
| FLAN-T5-large | 780M | Colab Pro | $10/mo | 48-52% |
| FLAN-T5-XL | 3B | Replicate/Modal API | $30-50 | 54-58% |
| mT0-large | 1.2B | Colab Pro | $10/mo | 52-56% |
| mT0-XL | 3.7B | API (zero-shot) | $1-5 | 56-60% |

#### Zero-Shot Evaluation (Too large to fine-tune)
| Model | Size | Cost | Expected |
|-------|------|------|----------|
| FLAN-T5-XXL | 11B | $10-20 | 56-60% |
| mT0-XXL | 13B | $5-10 | 58-62% |

---

### STAGE 3: MODERN DECODERS

#### Open-Source Decoders (Can self-host)
| Model | Size | Access | Cost | Expected |
|-------|------|--------|------|----------|
| Llama 3.1 8B Instruct | 8B | Local/Colab | $0 | 52-56% |
| Llama 3.1 70B Instruct | 70B | Together.ai API | $1-5 | 58-62% |
| Mistral 7B Instruct | 7B | Local/Colab | $0 | 50-54% |
| Mixtral 8x7B Instruct | 47B (sparse) | Together.ai API | $2-5 | 56-60% |

#### Proprietary Decoders (Cloud only)
| Model | Provider | Cost (982 test examples) | Expected |
|-------|----------|--------------------------|----------|
| Gemini 1.5 Flash | Google | $0 (free tier!) | 60-64% |
| GPT-4o-mini | OpenAI | $0.15 | 62-66% |
| Claude 3.5 Haiku | Anthropic | $0.80 | 60-64% |
| GPT-4o | OpenAI | $2-3 | 66-70% |

---

## NOTEBOOK STRUCTURE

```
models/location-models/

### PHASE 1: BASE MODEL SCREENING (Week 1)
├── 01_baselines.ipynb                    # Dictionary, SpaCy, Regex (30 min)
├── 02_seq2seq_base.ipynb                 # T5, FLAN-T5, mT5, mT0 (12 hrs)
├── 03_indian_seq2seq_base.ipynb          # IndicBART (3 hrs)
├── 04_bert_base.ipynb                    # BERT-base, confliBERT, MuRIL, IndicBERT, XLM-R (8 hrs)
└── 05_stage1_analysis.ipynb              # Select top performers (1 hr)

### PHASE 2: SCALING STUDY (Week 2, Optional)
├── 06_scaling_seq2seq.ipynb              # Fine-tune -large/-XL (6-8 hrs)
├── 07_scaling_zeroshot.ipynb             # Zero-shot -XXL (1 hr)
└── 08_stage2_analysis.ipynb              # Scaling curves (1 hr)

### PHASE 3: DECODER COMPARISON (Week 3, Optional)
├── 09_oss_decoders.ipynb                 # Llama, Mistral, Mixtral (2 hrs)
├── 10_proprietary_apis.ipynb             # Gemini, GPT-4o-mini, Claude (1 hr)
└── 11_final_analysis.ipynb               # Complete comparison (2 hrs)

### SHARED UTILITIES
└── utils/
    ├── data_utils.py                     # Data loading functions
    ├── evaluation_metrics.py             # compute_detailed_metrics()
    └── model_utils.py                    # Training helpers
```

---

## ACTION PLAN: WEEK 1 (For Next Week's Presentation)

### PRIORITY 1: Essential for Presentation (Can complete by Friday)

**Day 1-2 (Mon-Tue): Baselines + Initial Seq2seq**
- [ ] Create `01_baselines.ipynb` (30 min to implement, 30 min to run)
  - Dictionary lookup baseline
  - SpaCy NER baseline
  - Save results: `baseline_results.csv`

- [ ] Run `02_seq2seq_base.ipynb` - FLAN-T5-base (3 hours)
  - Already have T5-base (40%)
  - Add FLAN-T5-base section
  - Save results: `flan-t5-base_test_metrics.csv`, `flan-t5-base_test_predictions.csv`

**Day 3 (Wed): Multilingual Seq2seq**
- [ ] Continue `02_seq2seq_base.ipynb` - mT0-base (3 hours)
  - Test best multilingual + instruction-tuned model
  - Save results: `mt0-base_test_metrics.csv`

**Day 4 (Thu): BERT Models + APIs**
- [ ] Create `04_bert_base.ipynb` - confliBERT (3 hours)
  - **PRIORITY**: confliBERT for domain-specific comparison
  - Shows if conflict pre-training helps vs. multilingual pre-training
  - Save results: `conflibert_test_metrics.csv`

- [ ] Create `10_proprietary_apis.ipynb` (2 hours total)
  - Gemini 1.5 Flash (FREE via Google AI Studio)
  - GPT-4o-mini ($0.15 for test set)
  - Save results: `gemini-flash_test_predictions.csv`, `gpt4o-mini_test_predictions.csv`

**Day 5 (Fri): Analysis & Presentation Prep**
- [ ] Create comparison table and visualizations
- [ ] Key results for presentation:
  - Baseline: ~25%
  - Your best base model: ~50%
  - confliBERT: ~48% (shows domain vs. language trade-off)
  - API upper bound: ~62-65%
  - Privacy-performance trade-off curve

**DELIVERABLE FOR PRESENTATION**:
- Working comparison of 6-7 models including confliBERT
- Clear narrative: "Domain-specific (conflict) vs. language-specific (Indian) pre-training"
- Visual: Performance vs. Privacy plot with confliBERT highlighted

---

### PRIORITY 2: If Time Allows (Optional enhancements)

**Day 3-4: Add More Base Models**
- [ ] IndicBART (adds Indian language perspective)
- [ ] MuRIL-base (Google's Indian BERT, complements confliBERT)
- [ ] BERT-base (standard baseline to compare confliBERT against)

**Day 4-5: Quick Scaling Test**
- [ ] Zero-shot FLAN-T5-XL or mT0-XL via API ($5-10)
  - Shows what's possible without fine-tuning large models
  - Bridges gap between base and APIs

---

## EXPECTED RESULTS TABLE (For Presentation)

| Model | Type | Size | Exact Match | Privacy | Cost | Special Feature |
|-------|------|------|-------------|---------|------|-----------------|
| Dictionary | Rule-based | - | 25% | ✅ Full | $0 | - |
| SpaCy | Pre-trained NER | - | 20% | ✅ Full | $0 | - |
| **T5-base** | Seq2seq | 220M | 40% | ✅ Full | $0 | Standard baseline |
| **FLAN-T5-base** | Seq2seq | 250M | 45% | ✅ Full | $0 | Instruction-tuned |
| **mT0-base** | Seq2seq | 580M | 50% | ✅ Full | $0 | Multi + instruction |
| **confliBERT** | BERT | 110M | 48% | ✅ Full | $0 | **Conflict domain** |
| **Gemini Flash** | Proprietary | ? | 62% | ❌ None | $0 | Zero-shot |
| **GPT-4o-mini** | Proprietary | ? | 64% | ❌ None | $0.15 | Zero-shot |

---

## KEY TALKING POINTS FOR PRESENTATION

1. **Problem**: Extracting structured location data from Indian incident reports with spelling 
   variants and transliteration issues

2. **Challenge**: Privacy-sensitive data (can't send to commercial APIs) vs. need for performance

3. **Approach**: Three-stage evaluation (base → scaling → decoders)

4. **Week 1 Results**: 
   - Simple methods: 20-25% (insufficient)
   - Fine-tuned base models: 40-50% (acceptable, fully private)
   - **confliBERT**: ~48% (domain-specific pre-training helps!)
   - API models: 60-68% (best, but no privacy)

5. **Key Finding**: 
   - 10-15% performance gap for full data privacy - acceptable trade-off
   - Domain-specific pre-training (conflict) competitive with language-specific (Indian)
   - Suggests combining both approaches (conflict + multilingual) could be optimal

6. **Future Work**: 
   - Stage 2: Scale best performers (mT0-large, FLAN-T5-XL)
   - Stage 3: Comprehensive decoder comparison
   - **Novel contribution**: Fine-tune multilingual model on conflict data (mT5 + conflict corpus)
   - Expected: Close gap to ~55-58% while maintaining privacy

---

## RESEARCH CONTRIBUTIONS

### Novel Comparisons
1. **Domain vs. Language Pre-training**:
   - confliBERT (conflict domain) vs. MuRIL (Indian languages)
   - Shows which matters more for this task

2. **Privacy-Performance Frontier**:
   - Maps achievable performance at different privacy levels
   - Actionable for organizations with data sensitivity requirements

3. **Architecture Comparison**:
   - Seq2seq (T5/BART) vs. BERT for structured extraction
   - Both use same data, fair comparison

### Potential Future Work
- **ConfliBERT + Multilingual**: Fine-tune confliBERT on Indian text
- **Conflict-T5**: Create conflict-domain T5 (like confliBERT but seq2seq)
- **Hybrid approach**: Use confliBERT for entity detection, T5 for structuring

---

## REPRODUCIBILITY CHECKLIST

- [ ] All models use same train/val/test split (temporal: 80/10/10)
- [ ] All results saved with standardized naming: `{model-name}_test_metrics.csv`
- [ ] Same evaluation metrics for all (exact match, fuzzy match, per-field F1)
- [ ] Document random seeds, training hyperparameters
- [ ] Save all predictions for error analysis
- [ ] Code available in GitHub repo
- [ ] Note confliBERT version and pre-training corpus details

---

## CONTINGENCY PLANS

**If Colab crashes during training**:
- Each notebook saves checkpoints independently
- Can restart individual notebook without affecting others

**If running out of time**:
- Minimum viable: Baselines + 2 base models + confliBERT + 1 API = publishable
- Can add more models post-presentation

**If models don't fit in memory**:
- Use gradient checkpointing
- Reduce batch size, increase gradient accumulation
- Fall back to smaller model variants

**If confliBERT underperforms**:
- Still valuable negative result: domain pre-training alone insufficient
- Motivates combining domain + language pre-training

---

## SUCCESS CRITERIA

**For next week's presentation** (Minimum):
- ✅ 2-3 base models evaluated (T5-base, FLAN-T5-base, mT0-base)
- ✅ confliBERT evaluated (domain-specific comparison)
- ✅ 1-2 API comparisons (Gemini Flash, GPT-4o-mini)
- ✅ Baseline comparisons (dictionary, SpaCy)
- ✅ Clear privacy-performance narrative
- ✅ Domain vs. language pre-training insights

**For full paper** (Long-term):
- ✅ Complete Stage 1 (all base models)
- ✅ Stage 2 scaling study (1-2 large models)
- ✅ Stage 3 decoder comparison
- ✅ Error analysis and case studies
- ✅ Deployment recommendations
- ✅ Analysis of confliBERT performance on Indian conflict data

---

## ADDITIONAL NOTES

### About confliBERT
- **Paper**: "ConfliBERT: A Pre-trained Language Model for Political Conflict and Violence" (NAACL 2022)
- **Training Data**: Large corpus of political conflict and violence texts
- **Strengths**: 
  - Understands conflict-related language patterns
  - Better at conflict event detection and classification
  - Pre-trained on similar domain to your data
- **Considerations**:
  - Trained primarily on English text
  - May not handle Indian language transliterations as well as MuRIL
  - Interesting test: domain expertise vs. linguistic expertise

### Why No Conflict-Specific Seq2seq?
- confliBERT exists for BERT architecture only
- No "confliT5" or conflict-specific seq2seq model found
- **Research opportunity**: You could create one!
  - Take T5 or FLAN-T5
  - Continue pre-training on conflict texts (ACLED, UCDP, etc.)
  - Potential paper contribution

---

## ALTERNATIVE APPROACH: SIMPLIFIED EXTRACTION FOR GEOCODING

### THE PRACTICAL QUESTION
**"Do we really need structured hierarchical extraction, or would simple place name extraction work just as well for geocoding?"**

### MOTIVATION
Based on preliminary geocoding experiments, the Google Geocoding API appears to perform **equally well or better** with:
- **Simple approach**: Comma-delimited list of place names (e.g., "Bhadrachalam, Khammam, Andhra Pradesh")
- vs. **Complex approach**: Component filtering with structured hierarchy (state=X, district=Y, village=Z)

This raises critical questions:
1. **Is fine-tuning necessary?** Maybe zero-shot "extract place names" is sufficient
2. **Is structure necessary?** Maybe we're over-engineering the solution
3. **What's the actual end goal?** If it's geocoding, we should optimize for that, not F1 scores

### COMPARISON: STRUCTURED vs. SIMPLE

#### Current Approach (Structured Hierarchical)
```python
Prompt: "Extract location hierarchy from incident: {text}
Format: state: <name>, district: <name>, village: <name>, other_locations: <name>"

Output: "state: Andhra Pradesh, district: Khammam, village: Bhadrachalam, other_locations: Dornapal"

Evaluation: Exact match on all 4 fields (strict)
Problem: Penalizes if village/other_locations are swapped but geocoding would still work
```

**Advantages:**
- ✅ Interpretable (clear which level each place is)
- ✅ Can analyze performance by administrative level
- ✅ Enables filtering/validation (e.g., check if district is in state)
- ✅ Useful for non-geocoding downstream tasks

**Disadvantages:**
- ❌ Requires fine-tuning to learn structure
- ❌ More complex prompt
- ❌ Stricter evaluation (lower scores even if geocoding works)
- ❌ More work to implement

#### Alternative Approach (Simple Place Name List)
```python
Prompt: "Extract all place names from this incident: {text}"

Output: "Bhadrachalam, Khammam, Dornapal, Andhra Pradesh"

Evaluation: Did it geocode correctly? (end-to-end)
```

**Advantages:**
- ✅ Simpler prompt (better zero-shot performance)
- ✅ Order doesn't matter (more flexible)
- ✅ May not need fine-tuning at all
- ✅ Faster to implement and run
- ✅ **Optimizes for actual goal** (successful geocoding)

**Disadvantages:**
- ❌ No explicit hierarchy information
- ❌ Harder to debug failures
- ❌ Can't analyze by administrative level
- ❌ May include irrelevant place names

### PROPOSED EXPERIMENTS

#### Experiment 1: Zero-Shot Simple Extraction (2 hours, ~$0-1)
```python
# Use best zero-shot models
models = ['Gemini Flash', 'GPT-4o-mini', 'Llama 3.1 70B']

prompt = """Extract all place names mentioned in this incident report.
Return as a comma-separated list.

Incident: {incident_text}

Place names:"""

# Evaluate:
1. How many place names extracted?
2. Geocoding success rate (send to Google API)
3. Geocoding accuracy (distance from ground truth)
4. Cost per successful geocode
```

#### Experiment 2: Fine-tuned vs. Zero-Shot (compare both approaches)
```python
# Compare 4 scenarios:
1. Fine-tuned structured (current approach) → Google API
2. Fine-tuned simple list → Google API  
3. Zero-shot structured (GPT-4o) → Google API
4. Zero-shot simple list (GPT-4o) → Google API

# Metrics:
- Extraction F1 (structured only)
- Geocoding success rate (all)
- Geocoding accuracy in km (all)
- Cost per incident (all)
- End-to-end time (all)
```

#### Experiment 3: Ablation Study
```python
# Test Google Geocoding API with different input formats:

Input A (structured components):
  address="Bhadrachalam"
  components="locality:Khammam|administrative_area:Andhra Pradesh"

Input B (comma-delimited):
  address="Bhadrachalam, Khammam, Andhra Pradesh"

Input C (natural language):
  address="Bhadrachalam village in Khammam district, Andhra Pradesh"

# Measure:
- Which format gives best geocoding results?
- Does Google's API even use component filtering effectively?
```

### POTENTIAL FINDINGS

#### Scenario A: Simple Approach Works Just as Well
```
If zero-shot simple extraction + Google API achieves:
- 90%+ geocoding success rate
- <5km median error
- Comparable to fine-tuned structured approach

Then: Save weeks of work! Just use zero-shot APIs + Google Geocoding
```

#### Scenario B: Structure Adds Value
```
If fine-tuned structured approach achieves:
- Significantly better geocoding (>10% improvement)
- Better disambiguation (fewer wrong locations)
- More reliable results

Then: Structure is worth the complexity
```

#### Scenario C: Hybrid is Best
```
If combination works best:
- Extract simple list (zero-shot, cheap)
- Post-process with structure model (fine-tuned, selective)
- Use structured output for ambiguous cases only

Then: Two-stage pipeline optimizes cost/accuracy
```

### IMPLEMENTATION PLAN

#### Quick Test (This Week - 2 hours)
```python
# Notebook: 12_simple_extraction_pilot.ipynb

1. Take 100 random test examples
2. Run through Gemini Flash with simple prompt
3. Send results to Google Geocoding API
4. Measure success rate + accuracy
5. Compare to current structured approach

Cost: $0 (Gemini free tier)
Time: 2 hours
Outcome: Go/no-go decision on full experiment
```

#### Full Comparison (If Pilot Succeeds - 1 day)
```python
# Notebook: 13_structured_vs_simple_full.ipynb

1. Full test set (982 examples)
2. Compare all approaches (structured vs. simple, fine-tuned vs. zero-shot)
3. Measure end-to-end geocoding performance
4. Cost-benefit analysis
5. Recommendation for deployment

Cost: $5-10
Time: 1 day
Outcome: Paper contribution + deployment recommendation
```

### RESEARCH CONTRIBUTIONS

#### If Simple Works:
- **Negative result** (valuable!): "Complex structure unnecessary for geocoding task"
- **Practical impact**: Saves researchers from over-engineering
- **Generalizable**: Applies to other end-to-end extraction+geocoding pipelines

#### If Structure Matters:
- **Validates** current approach
- **Explains** when and why structure is necessary
- **Informs** future work on structured information extraction

#### Either Way:
- **Task-driven evaluation**: Optimize for end goal (geocoding), not proxy metrics (F1)
- **Cost-benefit analysis**: What's the ROI of fine-tuning?
- **Deployment recommendation**: What should practitioners actually use?

### KEY QUESTIONS TO ANSWER

1. **Does Google's API actually use component filtering effectively?**
   - Or does it just do full-text search regardless of components?

2. **How much does extraction quality matter?**
   - Is 80% extraction + good API = 90% extraction + same API?

3. **What's the cost-performance trade-off?**
   ```
   Option A: Zero-shot simple ($0.0002/incident) vs.
   Option B: Fine-tuned structured ($0 after training)
   
   If Option A works 90% as well, maybe worth the ongoing cost?
   ```

4. **What about disambiguation?**
   - Multiple places with same name (common in India)
   - Does structure help Google API disambiguate better?

5. **What about edge cases?**
   - When does structure help? (ambiguous locations, missing context)
   - When is simple enough? (clear location mentions, unique names)

### RECOMMENDED PRIORITY

**Before next week's presentation:**
- [ ] **Optional Quick Test** (if time Friday afternoon): 100 examples, simple extraction
- [ ] Mention in "Future Work" slide

**After presentation, before writing paper:**
- [ ] **Full comparison experiment** (1 day)
- [ ] This could be a **major contribution**: "Task-driven evaluation shows simple extraction sufficient for geocoding, saving significant engineering effort"

**For paper:**
- Section 4.3: "Alternative Approach: Simple Extraction for Geocoding"
- Compare extraction metrics vs. end-to-end geocoding performance
- Discuss implications for practitioners

### PRAGMATIC CONSIDERATIONS

#### Reasons to Prioritize This
1. **High impact if true**: Could simplify entire pipeline
2. **Quick to test**: 2-hour pilot answers key question
3. **Practical value**: Researchers want simple solutions that work
4. **Novel contribution**: Few papers compare proxy metrics vs. end-task performance

#### Reasons to Keep Current Approach
1. **Already invested**: Fine-tuned models working
2. **More general**: Structure useful beyond geocoding
3. **Better for analysis**: Can break down performance by admin level
4. **Academic contribution**: Model comparison still valuable

#### Hybrid Strategy (Recommended)
1. **Complete current plan**: Finish model comparisons (academic contribution)
2. **Add simple baseline**: Compare zero-shot simple extraction (practical contribution)
3. **End-to-end evaluation**: Report both extraction F1 and geocoding accuracy
4. **Best of both worlds**: Academic rigor + practical applicability

---

## FINAL NOTES

This alternative approach highlights an important research question: **Are we optimizing the right metric?**

If the end goal is geocoding, we should:
1. Report geocoding success rate alongside extraction F1
2. Consider whether fine-tuning is necessary
3. Evaluate cost-performance trade-offs
4. Provide practical deployment recommendations

The structured hierarchical approach may still be valuable for:
- Interpretability and debugging
- Non-geocoding downstream tasks (filtering by state, analyzing by district)
- Cases where Google API is not available
- Understanding model capabilities

But it's worth testing whether a simpler approach achieves similar **end-to-end** performance with less engineering effort.

**Bottom line**: Quick 2-hour pilot can answer whether we're over-engineering the solution. If simple works, it's a major contribution. If not, it validates the current complex approach. Either way, it's valuable!

---

END OF DOCUMENT

