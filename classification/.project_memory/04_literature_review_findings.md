# Literature Review: Threshold Formulation & Demographic Bias in Conjunctiva-Based Anemia Detection

Compiled 2026-07-28, in response to the observed `India_acc=0.43` vs `Italy_acc=0.88` gap. Reviews 4 documents placed in `D:\khaje\EYES-DEFY-ANEMIA\Source\Paper_Eye\` (a directory outside the tracked repo — see the new git-security rule in `CLAUDE.md` and `03_tech_stack_and_rules.md`).

**Important note on sourcing, read this before the findings below.** One of the four documents — `Anemia Detection Demographic Bias Research.docx`, the project author's own research summary — was cross-checked against the three primary-source PDFs during this review and found to contain real, material inaccuracies (detailed in the [Source Reliability](#source-reliability-a-necessary-caveat) section at the end). Everything in the sections below is labeled by provenance: **[Verified]** means I read the claim directly in the cited primary-source PDF myself; **[Docx only]** means it comes from the summary document and I could not independently check it (I don't have the underlying primary source — this applies to the Dimauro et al. and Asare et al. papers, neither of which was provided as a PDF). Treat `[Docx only]` claims with real caution given what was found below — one is independently corroborated by a second primary source (noted where it applies), the rest are not.

The three primary-source PDFs, referred to by first author throughout:
- **Paul et al. 2026** — "Conjunctiva Image Analysis for Anemia Detection: Evaluation on Eyes-Defy & CP-AnemiC," IEEE QPAIN 2026.
- **Sehar et al. 2025** — "Deep Learning Model-Based Detection of Anemia from Conjunctiva Images," Healthcare Informatics Research.
- **Ramos-Soto et al. 2025** — "Non-invasive anemia detection from conjunctiva and sclera images using vision transformer with attention map explainability," Scientific Reports (Nature Portfolio).

**Major finding up front:** Ramos-Soto et al. uses the **exact same Eyes-Defy-Anemia dataset as this project** — 217/218 images, India 95, Italy 122/123, one Italian patient excluded for missing Hgb — and applies genuinely country-and-gender-specific thresholds. This is the most directly relevant single fact in this whole review; see §1.

---

## 1. Target Formulation (Thresholds)

| Paper | Threshold used | Provenance |
|---|---|---|
| **This project (current)** | WHO: M<13.0, F<12.0, same both countries | — |
| **Ramos-Soto et al.** | **India: F<12.0, M<14.0. Italy: flat <10.5 (no gender data available).** | **[Verified]** — direct quote below |
| **Paul et al.** | Flat **<11.0 g/dL**, both countries, no gender split | **[Verified]** |
| **Sehar et al.** | No fixed clinical cutoff — Hgb is *regressed* from RGB values via a linear formula, then compared to "standard values" (not stated precisely in the paper) | **[Verified]** |
| Dimauro et al. (Eyes-Defy dataset's original paper) | <10.5 g/dL (Italy) | **[Docx only, but independently corroborated]** — Ramos-Soto et al. explicitly attributes their own 10.5 g/dL Italy threshold to "the dataset authors in previous studies," i.e. Dimauro et al. Two independent primary sources agree on this number even though I haven't read Dimauro et al.'s own paper. |
| Asare et al. | Adjusted/unspecified threshold, described only as avoiding "the noise associated with the WHO boundaries" | **[Docx only]** — I don't have this paper; treat as unverified. |

**Ramos-Soto et al.'s exact reasoning (quoted, not paraphrased), since this is the single most load-bearing fact in this review:**

> "Because pregnancy status was unavailable, sex-specific thresholds were applied with a screening orientation. For the Indian subset, an Hgb < 12 g/dL threshold was used for women, while an Hgb < 14 g/dL threshold was used for men. The 12.0 g/dL cutoff for non-pregnant women follows WHO guidance... For men, 14 g/dL is consistent with recent imaging-based screening studies (e.g., non-contrast CT) that prioritize sensitivity when flagging suspected anemia for confirmatory testing. For the Italian patient set, a threshold of Hgb < 10.5 g/dL is used, as determined by the dataset authors in previous studies, since no gender information was provided for this set."

This directly bears on a decision already made once in this project. `02_current_status.md` records that country/gender-specific thresholds (India F<12.0/M<14.0, Italy flat<10.5 — **the exact same numbers**) were proposed early in the classification module's history and rejected as "unsourced." That check was against the dataset's *own* documentation (`Dataset anemia.docx`), which is correct as far as it went — but it turns out these numbers **do have a real source**, just not one that was checked at the time: a peer-reviewed Scientific Reports paper analyzing this exact dataset. Worth knowing, even though — see §4 — the numbers turn out not to solve the problem they'd be adopted for.

**Why deviate from WHO at all?** All three papers give a real, physiologically-grounded reason, not just "our numbers work better": conjunctival pallor is a continuous optical gradient, and a patient at 11.9 g/dL (labeled anemic under WHO's female threshold) is visually indistinguishable from one at 12.1 g/dL (labeled non-anemic). A strict clinical cutoff placed inside that continuous, low-separability region forces a classifier to resolve two optically identical images into opposite labels — a real source of label noise for a *vision* task specifically, distinct from the imbalance question. Paul et al.'s and Ramos-Soto's choices both push the cutoff away from the WHO boundary for this reason (Paul et al. explicitly to reduce "marginal visual noise at the decision boundary"; Ramos-Soto's female threshold stays at the WHO line but their male threshold and Italy's flat threshold both move).

---

## 2. Handling Demographic & Domain Bias

**Only one of the three papers actually tests cross-country generalization with a stratified result — and even that one doesn't show anything close to an 88%/42% collapse.**

**Paul et al. [Verified] — the most directly relevant paper for this specific question:**
- Within-country accuracy (RUSBoost, Table IV): **India 0.729, Italy 0.838, Joint 0.813.**
- True cross-country transfer, training on one country and testing on the other with *no* retraining (Table V): **Italy→India 0.733, India→Italy 0.844.** Notably, this is barely worse than the within-country numbers — their pipeline generalizes across the India/Italy split reasonably well as-is.
- Their bias-mitigation techniques were HSV+CLAHE illumination normalization (applied to *both* countries before feature extraction) and a compact, interpretable 24-dimensional handcrafted feature set (RGB/HSV/CIELAB means+stds, entropy, and a "High Hue Ratio" redness ratio) rather than raw CNN pixels — the hypothesis being that handcrafted, aggregate color-ratio features are less prone to fixating on incidental local texture (skin/sclera pigmentation at the tissue boundary) than a CNN's learned convolutional filters are.
- **CORAL (Correlation Alignment)** was used, but for the harder **cross-*dataset*** transfer — Eyes-Defy (India+Italy) → CP-AnemiC (Ghana), a genuinely different population/dataset/device — not for India vs. Italy specifically, which didn't show enough degradation to need it. CORAL aligns the covariance matrices of source and target feature distributions in the 24-feature space; it raised the Eyes-Defy→CP-AnemiC balanced accuracy from a near-chance baseline to **0.556**.

**Sehar et al. [Verified] — does NOT address India/Italy or any demographic bias at all.** This is a single-cohort study (764 images, no country/region field in the dataset table, both authors based in Chennai). DCGAN augmentation (764→4,315 images) is used purely as a generic overfitting countermeasure for a small dataset — the paper does not frame it as a demographic-bias or ethnic-phenotype mitigation technique anywhere in its Methods or Results. The paper's own Discussion section explicitly flags this as a *limitation*, not something solved: "Future data collection efforts should focus on a diverse range of demographics to improve the model's generalizability across different groups." The docx's characterization of this paper as addressing "the threat of demographic overfitting" is not supported by the paper itself — see §5.

**Ramos-Soto et al. [Verified] — does not report a country-stratified accuracy split at all.** India and Italy are pooled into one combined 217-image dataset before the 80/20 train-test split; the paper's headline 98.47% accuracy is on that pooled set, not India-only or Italy-only. This means it's **not directly comparable** to our `India_acc`/`Italy_acc` metrics without first pooling our own predictions the same way. Their own Limitations section is candid about this: "the dataset is restricted to Italian and Indian cohorts, which may not fully capture ethnic, demographic, and acquisition variability. The proposed ViT-based framework has not yet been tested on external or real-world datasets... the present work should be interpreted as a proof-of-concept demonstration." Their attention-map analysis (below, §3) demonstrates *general* interpretability — attention lands on conjunctiva/sclera vascular regions rather than background — not a *measured* fix for a specific country-accuracy gap, since no such gap was ever computed in their evaluation.

**No paper in this set uses DCGAN, GAN-based synthesis, or any generative augmentation specifically to balance India vs. Italy representation.** (Sehar et al.'s DCGAN is for a different purpose entirely, as above.)

---

## 3. Methodology & Evaluation

| Paper | Models | Metrics reported | Best result |
|---|---|---|---|
| Paul et al. | RUSBoost, RF, Logistic Regression, SVM (RBF, tuned) — all on a 24-dim handcrafted feature vector, no deep learning | **Accuracy, Sensitivity, Specificity, Balanced Accuracy** — explicitly not accuracy alone (see quote below) | SVM(RBF) tuned on CP-AnemiC: 0.849 acc |
| Sehar et al. | SVM, KNN, Naïve Bayes, Decision Tree, GoogLeNet, Voting Ensemble, Stacking Ensemble (VGG16+ResNet-50+InceptionV3) | Accuracy, Precision, Recall, F1, AUC (ROC) | Stacking ensemble: 89.48% acc, AUC 0.97 (abstract states 0.97, not the higher-precision "0.972" the docx cites — see §5) |
| Ramos-Soto et al. | SVM, Naïve Bayes, XGBoost, Inception-V3, DenseNet-161, MobileNet-V2, ResNet-50, **ViT-B/16** (no-TL, ImageNet-1k TL, ImageNet-21k TL) | Precision/Recall/F1 per class, overall accuracy | ViT-B/16 (ImageNet-21k TL): **98.47%** accuracy, 1.00 precision / 0.97 recall (anemia), 0.97 precision / 1.00 recall (no anemia) |

**Paul et al. explicitly argue for reporting sensitivity/specificity, not just accuracy, in exactly the terms this project already cares about:** "The same work also compared the results with other classifiers... and emphasized the need to report sensitivity and specificity, not only accuracy, because anemia is often the smaller class" — worth noting this validates a decision already made in this project's v2 pipeline (sensitivity/specificity/balanced_accuracy added to `compute_metrics()` last session), independent of this literature review.

**A caveat worth carrying into any comparison with Ramos-Soto et al.'s 98.47%:** they explicitly did not run k-fold cross-validation or repeated splits — "Full k-fold cross-validation was not performed due to the small and demographically heterogeneous nature of the dataset... formal statistical significance testing was not conducted because each model was trained as a single deterministic run using a fixed train-test split." Their 98.47% is one run on one 80/20 split, not an averaged/cross-validated figure — this project's 12-trial Optuna search per (architecture, tissue_type) combo is a more rigorous protocol than what produced that number, which matters when the two get compared.

**Confirmed matching architecture detail:** Ramos-Soto et al.'s ViT-B/16 is described as "approximately 86 million parameters," 224×224 input, 12 encoder layers/12 heads — matches this project's own verified `vit_b_16` figures (86.57M params, 224 input) exactly. Same architecture, already in the v2 roster.

**A concrete, low-risk technique already validated by this literature, and already matching this project's existing choice:** Ramos-Soto et al. deliberately excluded color jitter, brightness scaling, and elastic deformation from their augmentation pipeline, "as these could artificially distort the subtle chromatic and structural markers of anemia" — using only horizontal/vertical flips. This project's `classification/scripts/dataset.py` augmentation (`HorizontalFlip` + `Rotate`, no color jitter) is already aligned with this — worth citing as external validation of an existing choice, not a change to make.

---

## 4. Synthesis for Our Project

**The central empirical question — would either literature threshold actually narrow the India/Italy imbalance on our own data — has a clear, computed answer, not a guess.** I ran both alternative threshold schemes against our real `metadata.csv`:

| Threshold scheme | India anemic | Italy anemic | Gap |
|---|---|---|---|
| **Current (WHO M<13.0/F<12.0)** | 68/95 (71.6%) | 23/122 (18.9%) | **52.7 pp** |
| Ramos-Soto et al. (India F<12/M<14, Italy flat<10.5) | 79/95 (83.2%) | 11/122 (9.0%) | **74.1 pp** — worse |
| Paul et al. (flat <11.0, both countries) | 40/95 (42.1%) | 14/122 (11.5%) | **30.6 pp** — better |

**Ramos-Soto's exact thresholds — despite now having a real citable source — would make our specific imbalance problem worse, not better, on our data.** Raising India's male cutoff from 13.0 to 14.0 pulls *more* India men into the anemic class (India's mean male Hgb is 12.81, comfortably under 14.0), while Italy's flat 10.5 cutoff is far stricter than WHO's gendered 12.0/13.0 (Italy's mean Hgb is 13.83), pulling Italy's anemic rate down even further. This is the same directional effect the original "unsourced thresholds" decision predicted back when it was rejected — now confirmed quantitatively, and with the added context that the thresholds *are* real (just not helpful for this specific goal).

**Paul et al.'s flat 11.0 g/dL threshold is the one alternative that empirically helps**, cutting the gap from 52.7pp to 30.6pp and bringing India's label distribution close to balanced (42.1% anemic) — worth taking seriously as an actual candidate, not just literature color. Trade-off to weigh explicitly: it relabels a real number of patients in the 11.0–13.0 g/dL range (borderline under WHO, non-anemic under this scheme) and was adopted by Paul et al. for cross-*dataset* comparability with a pediatric-threshold-convention dataset (Ghana/CP-AnemiC), not because 11.0 is independently the "right" adult cutoff — that provenance is worth being explicit about if this gets adopted, not just citing the accuracy improvement.

**Concrete, well-grounded techniques from this literature that don't touch the label definition at all, in rough order of how directly they target the specific `India_acc`/`Italy_acc` gap:**

1. **Country-stratified evaluation is already the right lens — Paul et al.'s own framing reinforces this.** Their finding that within-country and true cross-country transfer accuracy are similar (0.729–0.844 range, no collapse) while still being lower than a from-a-different-domain naive expectation suggests illumination normalization + compact, interpretable features generalize across this exact India/Italy gap reasonably well *without* deep learning at all. Worth a real comparison point: does our CNN/ViT roster's `India_acc`/`Italy_acc` gap shrink if features are extracted through the same HSV+CLAHE normalization Paul et al. use, before any model sees the image?
2. **HSV+CLAHE (Value channel only) or CIELAB a\* -channel isolation as a preprocessing step**, applied identically to both countries before the existing pipeline — cited by Paul et al. as the actual mechanism that kept their India/Italy gap from becoming severe. This is additive to the existing `dataset.py` pipeline, not a replacement — worth prototyping as a preprocessing variant rather than a full retrain commitment.
3. **CORAL for the *within-Eyes-Defy* India↔Italy split**, not just cross-dataset — Paul et al. only applied it cross-*dataset* (Eyes-Defy→Ghana) because their India/Italy gap wasn't severe enough to need it. Ours may be, given the reported `India_acc=0.43`. This is a real, implementable technique (align the covariance of India vs. Italy feature distributions in the trained backbone's embedding space) that no paper in this set has actually tested for exactly this pairing — a genuine, defensible novel contribution for the thesis if it's tried here.
4. **`balanced_accuracy`/`sensitivity`/`specificity` as the reported headline metrics, not raw accuracy** — already implemented in this project's v2 pipeline, and independently reinforced by Paul et al.'s explicit argument for the same thing.

**What I would not recommend on this evidence:** switching to Ramos-Soto's exact country/gender thresholds specifically to fix the accuracy gap — the computed numbers above show it moves in the wrong direction for that specific goal, even though the thresholds themselves are now properly sourced.

---

## Source Reliability: a necessary caveat

The research summary docx was checked against the three PDFs it discusses, and contains real, specific inaccuracies worth knowing about before relying on it further:

1. **The document's own opening hook — "an 88% diagnostic accuracy on a European (Italian) cohort while degrading to a 42% accuracy on a South Asian (Indian) cohort... highlights the acute vulnerability of standard neural networks"** — is presented as if it's a documented literature finding. It isn't. No paper discussed anywhere in the same document reports that specific pairing; the closest real number (Paul et al.) is 0.838/0.729, not 0.88/0.42. This pairing is suspiciously close to this project's own `India_acc=0.4286`/`Italy_acc=0.8824` reported two turns before this document was produced. I can't prove causation, but I can confirm the number doesn't trace to any cited source in the document that contains it — treat the document's framing device as unsupported, not as an independent literature confirmation of our own result.
2. **Sehar et al.'s DCGAN is mischaracterized.** The docx frames it as addressing "the threat of demographic overfitting... prevent[ing] the subsequent classification models from overfitting to a specific ethnic phenotype." The actual paper uses DCGAN purely as generic small-dataset augmentation and explicitly lists demographic diversity as unaddressed future work, not a solved problem.
3. **Ramos-Soto et al.'s domain-shift-solving claim is overstated.** The docx frames the ViT/attention-map result as demonstrated protection against the India/Italy accuracy gap. The actual paper never computes a country-stratified accuracy at all — its 98.47% is on a pooled dataset — so it cannot have demonstrated a fix for a gap it never measured.
4. **A minor false-precision addition**: the docx states Sehar et al.'s AUC as "0.972"; the paper (abstract and figure) states 0.97.
5. **Two of the five papers discussed (Dimauro et al., Asare et al.) were not provided as PDFs and are not independently verified here** — one (Dimauro et al.'s 10.5 g/dL Italy threshold) happens to be corroborated by a second real primary source; the other (Asare et al.) is not corroborated at all in this review.

None of this means the summary is worthless — its color-space and CORAL/domain-adaptation descriptions checked out accurately against Paul et al., and its citation list (titles, DOIs, journals) all resolved to real papers. But the specific framing that most directly motivated this review — "literature has documented this exact 88%/42% phenomenon" — did not hold up, and the report above is built from the primary sources directly rather than from that framing.
