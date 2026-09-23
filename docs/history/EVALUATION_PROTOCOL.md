> Historical snapshot. For the current setup and file layout, see the [project README](../../README.md). Earlier raw experiment outputs are preserved in [the archive](../../data/archive/early_experiments.zip), using their original paths. Audio and plot links may refer to local-only files.

# Evaluation protocol for the first music-detection experiments

Version: 0.2 — 21 September 2026.

**Status:** design document. No eligibility decisions, splits, downloads, training,
thresholds, or evaluation results are created by this document. Numerical choices
listed as pending must be fixed before the relevant experiment runs.

## 1. The question we want the experiment to answer

Can a classifier trained on our audio features distinguish historical FMA human
reference labels from Echoes TTA generated labels on eligible recordings whose
reference groups and human artists were not used to develop the classifier?

A separate experiment will ask whether it also works on a generator family absent
from training and validation. Neither experiment can establish universal detection,
prove authorship, or validate mixed production and microphone playback.

Our existing pipeline produces ten descriptive features from a ten-second mono
clip at 24 kHz. Those features are candidate predictors, not established AI
signatures. The six inspected recordings verify pipeline operation only.

## 2. Eligibility comes before split assignment

Maintain a review decision for each original recording: `eligible`, `needs_review`,
or `excluded`, with the evidence, reason, reviewer and date. These are proposed
fields for a future manifest; they are not new labels already applied to our data.

| Question | Evidence or rule |
|---|---|
| Where did this recording come from? | Stable source identifier, source URL, dataset revision and local content hash |
| Why is its human/AI label credible? | Documented provenance and production context; no inference from an absent AI label |
| What usage terms apply? | Track-specific license/source terms and attribution details; unresolved terms stay flagged |
| Does it fit our scope? | Music; credible human reference or documented fully text-generated audio; no ATA/hybrid examples initially |
| Is the content usable? | Decodable finite samples, sufficient duration, listening/content check and documented quality issues |
| Which other records are related? | Reference, artist, alternate-version and duplicate relationships |

### Approved research-label standard (21 September 2026)

The user approved an exploratory dataset-label scope: historical FMA recordings
serve as human references, and Echoes TTA recordings serve as generated examples.
These are evidence-supported working labels, not independently certified authorship.
Evaluation measures predictions against these dataset labels; possible label errors
must be disclosed in results and any article or model card.

Preserve historical source/artist evidence and all `provenance_status=unverified`
flags. Independent authorship certification is no longer an admission requirement
for this exploratory baseline. That flag alone does not exclude an otherwise usable
record, but contradictory evidence must still be investigated. Listening checks
content; it cannot certify authorship. No percentage of label certainty is assigned.

This approval establishes the label standard only. Per-record source consistency,
usage terms, audio integrity/content, quality and duplicate checks still determine
eligibility. No existing candidate is automatically marked eligible by this change.
User-supplied audio needs no source metadata; only extracted audio features enter
the classifier. Direct-file evaluation comes first; coffee-shop playback and
microphone validation remain deferred.

For generated recordings, preserve generator/model identifiers, generation mode,
dataset revision, and the source's generation record. A provider name alone may
not distinguish related model versions or aliases.

Current evidence in [data/audit.md](audit.md): 116 provisional human references,
109 with complete metadata, but all 116 still marked unverified for authorship.
Thus this document does not declare 109 records eligible. The seven legacy-license
cases, ambiguous names, repeated paths and NoDerivatives cases remain subject to
their recorded decisions; do not reinterpret those terms as automatic legal bans.

Straw Fields' large waveform offset remains a quality-review issue. Preserve its
original bytes and determine a policy before modeling. A repair, replacement or
exclusion must be logged and applied without looking at final-test performance.

## 3. Build groups before extracting experimental clips

A split unit is a connected group of related original recordings, not a CSV row
or an individual ten-second clip.

Join records when any of these relationships is known:

1. The same original human reference, including all generated counterparts.
2. The same human artist across references; generated counterparts inherit their
   reference's grouping relationship. This does not attribute the AI recording to
   the human artist.
3. Duplicate audio, alternate encodings, edits or related versions of a recording.

Apply these relationships transitively. If A shares a reference with B and B is
linked to C through an artist or duplicate relationship, all three stay together.
Each resulting group receives exactly one assignment. Do not split a large group
merely to achieve attractive class balance or target percentages.

Exact file hashes identify byte-identical duplicates. Different hashes do not
prove different audio. Before a larger experiment, define how alternate encodings
and near-duplicates will be screened and reviewed. If uncertain, merge the related
records conservatively or quarantine them pending review. Conflicting labels for
duplicate content must be investigated, not resolved by majority vote.

Missing artist IDs must not become one global 'unknown artist' group, nor be
treated as proof of distinct artists. Artist-disjoint evaluation requires adequate
artist evidence. If it cannot be supported, revise the protocol explicitly and
report the narrower reference-disjoint claim.

### Concrete example

Digital Lightning's human excerpt, its Udio counterpart and other counterparts
belong to one reference group. Another Cloudkicker reference links that group to
its own counterparts through the artist. None may appear on opposite sides of a
claimed artist-disjoint train/test split.

## 4. The inspected pilot is development data

The six files in [data/audio_pilot_manifest.csv](../../data/preparation/audio_pilot_manifest.csv)
have already influenced implementation and our understanding of confounds.
Keep their complete connected groups in the development pool, excluded from both
final-test sets. Discovering another related record later propagates that exclusion.

These files may support debugging or, if independently eligible, future training.
They cannot become an untouched test set by changing their filenames or using
different clip positions. The currently saved first-ten-second features are smoke
checks, not a training/test split.

## 5. Assign training, validation and test roles

| Role | Permitted use |
|---|---|
| Training groups | Fit model parameters, scalers and any learned feature transformations |
| Validation groups | Compare predefined models, select hyperparameters and operating thresholds |
| Final-test groups | Measure the frozen procedure once decisions are fixed |

Inventory eligible groups, artists, tracks, labels, genres and generators first.
Then choose feasible target counts/proportions and a fixed random seed. Record
both requested and actual counts. Group separation takes priority over balance.
Never try multiple seeds and retain the one with the best test score.

Training and validation must contain usable examples of both classes. Both final
evaluation conditions require human controls and AI examples. If the eligible
group count is too small, report feasibility limits and propose more data or a
clearly labeled exploratory grouped validation study. Do not manufacture a final
test claim from a few correlated examples.

Group-aware folds inside the development pool may be considered later if a single
validation split is unstable. A separately protected final test set remains distinct.
No percentages, minimum counts or fold counts are fixed in this version.

Training-class weighting or subsampling must be specified in advance and use
training data only. Do not duplicate test human tracks to balance numerous AI
counterparts. Report the naturally selected class and group counts, and keep the
evaluation cohort identical across models being compared.

## 6. Unseen-generator evaluation without reference leakage

Select and record one held-out generator family after checking eligible coverage
and model/provider aliases, but before fitting models. Its audio must appear in
neither training nor validation. Generator choice must not depend on test scores.

First reserve final-test reference/artist groups. Within those groups define:

- A seen-generator condition: eligible human controls plus eligible AI tracks from
  families available during training.
- An unseen-generator condition: eligible human controls plus eligible tracks from
  the held-out family.

Held-out-family outputs associated with training or validation groups are excluded
from modeling and final evaluation. Their existence does not make their references
eligible for the final test; record these unused rows and their reason.

The two final conditions may share human controls and reference groups. Report
them separately and disclose that dependence; do not combine them as independent
observations or count shared humans twice in one headline metric.

This design evaluates an unseen family on unseen reference/artist groups. It does
not isolate only generator effects. If paired seen/unseen coverage is incomplete,
report cohort differences and any matched-reference comparison separately.

If the held-out family is represented in the already inspected six-file pilot,
exclude those development groups as above and disclose that the family was seen
during engineering. Do not claim a completely blind generator evaluation. The
specific family remains pending in this protocol.

## 7. Freeze clipping, preprocessing and feature rules

Use the same preprocessing code and feature order for all classes and splits.
The current baseline representation is mono, 24 kHz, ten seconds, five frame
properties summarized by mean/std. See [features.py](../../features.py) and
[data/pilot_features_config.json](../../data/archive/early_experiments.zip).

Before experimental extraction, fix the number of clips per track, start-time
selection policy and short/silent-file policy. Apply them without class labels
or listening for a 'good' detector example. Compare equivalent clip durations and
coverage; do not give long AI files many more votes than human excerpts.

If multiple clips are used, every clip inherits its original group's split.
The proposed first aggregation is the arithmetic mean of clip-level AI-class
scores per original track, followed by a track-level decision threshold. Validate
and freeze that choice before final evaluation; it does not automatically produce
a calibrated track probability. One clip per track is a simpler alternative to
consider once coverage and runtime are known.

Feature columns are exactly FEATURE_NAMES. Labels, artist IDs, filenames, source,
generator and review notes are not model inputs. Any scaler, imputer, feature
selection or learned preprocessing fits on training data only. A future
scikit-learn Pipeline should bind scaling and classification together.

For a later encoder comparison, preserve the same source tracks, groups and time
intervals. Use the encoder's documented rate/preprocessing requirements rather
than forcing 24 kHz. Record the representation difference and fit its classifier
on training data only. Model choice and loading remain separate work.

## 8. Thresholds, inconclusive results and calibration

For the initial binary experiment, choose an operating threshold on validation
track scores under a declared objective, such as balancing recall against a
specified human false-positive limit. The numerical objective and fallback when
it cannot be met are pending; fix them before threshold selection.

For the intended three-outcome product, a later experiment may use a lower and
upper threshold: below the lower threshold gives likely human-made, above the
upper gives likely AI-generated, and the middle is inconclusive. Select and
freeze both using validation data. Neither an inconclusive option nor a low score
guarantees reliability on unfamiliar or mixed audio.

Raw classifier outputs must not be presented as calibrated confidence percentages.
If calibration is added, define grouped calibration/model-selection partitions or
an appropriate grouped development procedure first. Final-test data cannot fit
the calibrator or influence threshold choices.

## 9. Required measurements and their denominators

Treat AI as the positive class. Each original track contributes once per condition.
For a binary decision, report:

| Metric | Definition |
|---|---|
| AI precision | Correct AI predictions / all AI predictions |
| AI recall | Correct AI predictions / all truly AI-labeled tracks |
| Human false-positive rate | Human tracks predicted AI / all human tracks |
| Confusion matrix | True label by predicted label, with raw track counts |

Always include total tracks, class counts, independent groups/artists and per-family
counts. A zero denominator is reported as undefined, not as a perfect score.
Accuracy may be supplementary, but must not replace the above metrics.

For three outcomes, publish a 2-by-3 count table: true human/AI against predicted
human/AI/inconclusive. Count inconclusive AI tracks as not detected in overall AI
recall; keep inconclusive human tracks in the human false-positive denominator.
Also report overall/per-class inconclusive rates and coverage (fraction receiving
a conclusive decision). Any conclusive-only metric must be labeled as conditional
and shown alongside coverage; do not hide abstentions to improve apparent results.

For uncertainty intervals, resample independent connected groups rather than
correlated clips. Fix the interval method, seed and repetitions before final use;
retain class-count checks for resamples. With too few groups, show raw counts and
explain that interval estimates are unstable. Do not invent precision from a small
test set: one false positive among 20 human tracks changes FPR by five percentage points.

## 10. Confounds, robustness and errors

Before fitting, summarize class-specific sample rate, channels, duration, encoding,
bitrate where available, amplitude statistics, genre and source. Our six-file
pilot already shows rate/duration/amplitude differences and one large offset.
Resampling does not erase codec history, and RMS is not perceptual loudness.

Define any ablation (for example, removing energy-related features) as a separate
development experiment, not a test-driven repair. Where available, use additional
human sources and held-out generators to assess collection-specific effects.

Compression, noise, resampling and recorded-playback tests come later. Fix their
settings before evaluation and apply comparable transformations to both classes.
All variants inherit the original group. Do not count transformed copies as new
independent tracks or add final-test-derived audio to training augmentation.

Listen to representative validation errors during development, preserving notes
about false positives and false negatives. Inspect final-test errors only after
the frozen evaluation. A subsequent fix is a new development iteration; the old
test set is then exposed and cannot support an unchanged 'untouched test' claim.

Explanations must identify actual measured properties or tested model influence.
Do not infer synthetic vocals or causal AI artifacts from a score, centroid, or
highlighted segment alone.

## 11. Freeze record and future verification checks

Before final evaluation, save the eligible manifest and exclusions, source/model
revisions and hashes, grouping relationships, split assignments, seeds, clip
policy, preprocessing/feature settings, model/scaler configuration, aggregation,
thresholds, metrics and robustness settings. Preserve the development results
that motivated selections. Document any post-freeze change and its consequences.

The future split implementation should have behavioral tests proving that:

- Reference/artist/duplicate relationships propagate transitively to one group.
- No group crosses training, validation and final-test boundaries.
- All six pilot groups and later-discovered relatives stay out of final testing.
- Held-out-family audio is absent from training/validation, including known aliases.
- Unknown/conflicting eligibility never silently becomes an admitted label.
- Every admitted recording has a recorded assignment or explicit unused reason.
- Feature fitting only sees training groups; clip and variant splits inherit parents.
- The same frozen input manifest, configuration and seed reproduce assignments.
- Metrics count tracks correctly, handle empty denominators and retain abstentions.

Passing those checks proves the implementation follows the rules. Held-out
experiments will still be needed to establish whether the model is useful.

## 12. Decisions still required before implementation/evaluation

| Decision | Evidence needed |
|---|---|
| Eligible records under approved dataset-label standard | Complete per-record source/content and quality review; preserve evidence limits |
| Treatment of offsets and other quality issues | Source/listening checks and a class-consistent policy |
| Duplicate/alias review method | Available IDs, audio integrity evidence and manageable review workload |
| Split counts/proportions and seed | Eligible connected-group inventory and class/genre/generator coverage |
| Held-out family | Sufficient independent coverage and resolved provider/model aliases |
| Clip count and selection; aggregation confirmation | Duration/silence coverage and development feasibility |
| Model search and class balancing | Fixed development plan and available group counts |
| Threshold objective and calibration plan | Intended use and validation sample support |
| Interval and robustness settings | Independent group counts and a bounded experiment plan |

No dataset expansion, code implementation or training is authorized by this file.
For now its acceptance criterion is conceptual: we can explain eligibility, which
records must stay together, what each split is allowed to influence, and what the
resulting measurements would actually claim.

### Learning check

If two generated clips reference one human song, can one train the classifier
while the other evaluates it? Under this protocol, no: the shared reference links
their groups. If another song by the same artist is added, that relationship must
also propagate before assignment. Try sketching this example with three songs
and two artists before we implement a splitter.

Prepared with Codex assistance under the approved documentation-only scope.
