<img src="docs/nadi.svg" width="64" alt="NADI logo">

# SmartGuard JKN, powered by NADI

NADI is a small language model, trained from scratch on sequences of Indonesian national
health insurance (JKN) claims. It indicates which hospital claims a BPJS Kesehatan verifier
should review first. SmartGuard JKN is the web application built around the model.

Project page with an interactive demonstration (in Indonesian): https://aliifrahman.github.io/nadi/

The application on synthetic claims (in Indonesian): https://aliifrahman.github.io/nadi/app/

A one-minute overview video (in English, with Indonesian subtitles): https://aliifrahman.github.io/nadi/nadi-overview.mp4

This repository is an entry to the BPJS Kesehatan Healthkathon 2026, in the Health Facility
Risk Efficiency category (*Efisiensi Risiko Fasilitas Kesehatan*). It addresses upcoding,
episode splitting, repeat billing, and readmission.

## Overview

BPJS Kesehatan receives far more hospital claims than verifiers can examine individually, and
the claim data carry no indication of which claims are problematic. NADI addresses this by
ranking claims rather than judging them.

The approach is comparable to next-word prediction on a mobile keyboard. A keyboard can
suggest the next word because it has processed a large volume of text. NADI has processed
1.7 million claims, and it can therefore estimate the content of a member's next claim from
that member's claim history. A claim that differs substantially from the estimate is placed
near the top of the review queue.

For example, one member in the sample data was hospitalised for four days for a hip joint
procedure, and the hospital billed severity level III. NADI estimated severity level II for
81 of 100 comparable claims. The difference in tariff is approximately Rp16.7 million, so
the claim is ranked near the top of the queue.

NADI does not make decisions and does not allege fraud. The decision remains with the
verifier.

The model has 4 million parameters and was trained in 23 minutes on a single laptop. It uses
no pretrained model and no third-party API. Further details are provided in
[MODEL_CARD.md](MODEL_CARD.md) (in Indonesian).

## How it works

1. Each hospital (FKRTL) claim is converted into 12 tokens: sex and age group, days since the
   previous claim, same or different facility, outpatient or inpatient, hospital class,
   primary diagnosis, INA-CBG code, length of stay (LOS), three secondary diagnoses, and
   severity level.
2. A GPT-style model (4M parameters, trained from scratch) learns to predict the next token
   from the member's claim history. Training is self-supervised, so no fraud labels are
   required.
3. The probability that the model assigns to the claimed value serves as the score. For
   example, a claim billed at severity level III, which the model rates at 27% against 64% for
   severity level I, is prioritised for review.
4. Patterns that are already well defined, namely episode splitting, 30-day readmission,
   overlapping stays and repeat billing, are handled by rules.
5. At facility level, each facility's share of claims at severity level II or III is compared
   with that of its peers (same CBG and same hospital class) and displayed as a funnel plot.
   This comparison uses a frequency table rather than the model (see Limitations).
6. The verifier records a decision for each claim: eligible (*layak*), pending, or not
   eligible (*tidak layak*). Decisions are written to an audit log and will, in time, provide
   the labels that do not exist today.

The score sets the priority of review. It is not a decision.

## Evaluation

The data are the regular sets of the BPJS Kesehatan Sample Data, editions 2021 to 2025. Members
found in two editions have the same date of birth and sex in every case checked, so the editions
form one longitudinal cohort: 6,998,359 hospital claims from 756,440 members, discharged between
2019 and 2024.

The model was trained only on claims discharged up to 2023, and 10% of members were left out of
training altogether (4,757,638 training claims, 10,000 steps, 52 minutes on a MacBook M3 with
8 GB). It was then tested on the 195,544 inpatient claims of 2024, a year it never saw
(`validate.py`).

The sample data contain no fraud labels, and the organisers confirmed that no claim status data
are available. The evaluation therefore uses injected anomalies: claims are modified according
to a known pattern, and recall is measured among the 5% of claims with the highest scores.

| Test on 2024 claims | NADI | Baseline |
|---|---|---|
| Severity accuracy, every inpatient claim | 96.4% | 83.0% (table of CBG by number of secondary diagnoses) |
| Severity accuracy, members left out of training | 96.5% | 82.5% |
| Severity accuracy, members with no claim before 2024 | 96.6% | 83.9% |
| Severity raised from I to II, other fields unchanged | recall 97% | 71% |
| Realistic upcoding: one secondary diagnosis added, severity I to II | recall 50% | 18% |
| One inpatient episode split into two claims | recall 98% | rule: recall 100%, 11 false positives |
| LOS extended by 3 to 5 days (`evaluate.py`) | recall 47% | 44% (table of CBG by severity) |

Accuracy one year ahead matches the 96.2% that an earlier model reached within a single year, so
the model has learned coding patterns that hold across years. For realistic upcoding, NADI finds
far more cases among the top 5% of claims, but over the whole ranking the table is slightly
better (AUC 0.888 against 0.832).

To find half of the realistic upcoding, a verifier following NADI's order checks 5.2% to 6.2% of
inpatient claims, depending on the injection draw. The table needs 10.7%, and random order 50%.

Of the 195,544 inpatient claims of 2024, 3,428 have a severity level above the model's
prediction. The difference from the median tariff is approximately Rp11.6 billion in the sample.
With sampling weights applied, this corresponds to approximately Rp1.0 trillion per year, or
1.11% of inpatient cost. This figure is a rough estimate.

### Patterns in the real claims, 2019 to 2024

Rules applied to every claim of the six years produce review candidates:

| Pattern (sub-category) | Claims, 2019-2024 | Weighted, per year | Top 5% of facilities hold |
|---|---|---|---|
| Episode split: back at the same hospital with the same diagnosis within a day (8) | 417 | about 5,800 claims, Rp42 billion | 69% |
| Readmission within 2 to 30 days, same hospital and diagnosis, planned care excluded (16) | 12,388 | about 161,000 claims | 35% |
| Inpatient stay that starts before the previous stay has ended (11) | 372 | about 4,700 claims, Rp53 billion | 73% |
| Exact duplicate claim (11) | 110, of which 109 outpatient | about 1,200 claims | |

Planned care (chemotherapy and radiotherapy, dialysis, chronic kidney disease, thalassaemia,
cancer) made up more than half of all 30-day readmissions and is excluded from that rule.
Most remaining readmissions are likely to be medically justified. 92% of overlapping stays
involve two different hospitals, and the median overlap is one day, which fits a referral
recorded with a one-day offset. The 19% of overlapping stays that lie entirely within another
stay are the strongest candidates.

### Do the same facilities stand out every year?

Each facility's share of claims at severity level II or III is compared with its peers. For this
index (sub-category 4), the Spearman correlation between consecutive years is 0.66 to 0.71 across
1,081 to 1,679 facilities. Of the facilities above the 3σ limit in one year, 24% to 59% are above
it again the next year, against a base rate of 2%. In 2024, 33 of 1,906 facilities lie above the
limit.

Facilities with an episode split recur the next year in 4% to 31% of cases (base rate 2% to 3%),
and facilities with an overlapping stay in 14% to 19% (base rate 1% to 3%). A pattern that recurs
at the same facility is a practice to review, not chance. Recurrence alone does not separate
fraud from case mix or local coding habits.

### Tested and not used

- The difference between billed and verified amounts as a label. Over the five editions, 3,313
  inpatient claims (0.4%) were paid less than billed. The share is the same for severity I, II
  and III, the amount paid matches the tariff of a lower severity in about 3% of cases, and 76%
  of the cuts fall in facilities where nearly every claim was cut that year, at round ratios such
  as 0.7 and 0.9. This points to tariff rules rather than reviewer corrections.
- Blending the NADI and table scores. AUC rises to 0.890, but recall in the top 5% falls from
  48% to 42%.

The web application displays the 1,699,219 claims of the 2025 edition, all from a year the model
did not train on. It reads each member's history within that year only.

## Limitations

- Upcoding that is internally consistent at the coding level is detected in only about half
  of cases. The clinical evidence is in the medical record, not in the claim data.
- For prolonged LOS, NADI performs almost the same as the baseline (47% against 44%).
- The readmission rule is weak: facility rates barely persist between years (Spearman 0.14 to
  0.24 together with episode splits). Its hits are candidates only.
- Rupiah figures are differences from the median tariff based on the model's prediction.
  They are not audit findings.
- At facility level, NADI does not outperform a table. The expected severity derived from
  NADI, obtained by sampling secondary diagnoses, was tested and was no more stable between
  half-years than the table of CBG by hospital class (Spearman 0.56 against 0.63 in the same
  experiment). The table is therefore used.
- The facility ratio is a comparison, not evidence. Case mix that is not captured by CBG and
  hospital class may account for the difference.
- The sample data have no facility identifier column. The facility identifier is derived
  from the prefix of the visit identifier. This is an inference and is not documented
  officially.
- The application has no login or access control yet. It has not been tested by verifiers, and
  no pilot has been conducted.

## Reviewer decisions and audit log

Decisions are stored in `artifacts/reviews.jsonl`, which is append-only. Each line contains
the SHA-256 hash of the preceding line, so any alteration of an earlier line is detected.
The status of the hash chain is shown in the footer of the application. Each entry records
the time, the verifier's name, the decision, a note, and the probability given by the model
at the time of the decision.

This prototype has no login, so the verifier's name is entered manually. The hash chain does
not detect removal of the final line.

## FHIR and SATUSEHAT interoperability

| Endpoint | Purpose |
|---|---|
| `GET /api/claim/{row}/fhir` | The assessment as a Bundle: RiskAssessment, Provenance, and Device |
| `GET /api/claim/{row}/fhir/claim` | The claim and its history as a Bundle: Patient, Organization, Encounter, and Claim |
| `POST /api/fhir/assess` | Accepts a claim Bundle and returns an assessment Bundle. With `?send=true`, the result is also delivered |
| `GET /api/claim/{row}/satusehat` | Preview of the request to SATUSEHAT. Nothing is sent |

Within the SATUSEHAT claim flow, NADI operates on the BPJS Kesehatan side, after the `Claim`
is received and before the `ClaimResponse` carrying the verification result is issued.

The delivery target is set through environment variables, never through a request:

```bash
# SATUSEHAT (OAuth2 client credentials). The default host is the sandbox.
export SATUSEHAT_CLIENT_ID=... SATUSEHAT_CLIENT_SECRET=...
export SATUSEHAT_HOST=https://api-satusehat.kemkes.go.id   # optional, for production

# or any other FHIR R4 server
export FHIR_BASE=http://localhost:8080/fhir
```

If neither is set, the application runs in dry-run mode: the request is displayed and not
sent. Only the `RiskAssessment` is sent to SATUSEHAT, because `Provenance` and `Device` are
not yet supported there. For other FHIR servers, the claim and its assessment are sent as a
single transaction. A resource that carries an identifier is not created a second time when
it is sent again.

Claims from the BPJS Kesehatan sample data can only be previewed. Actual delivery applies
only to Bundles received through `POST /api/fhir/assess`. `FHIR_BASE` must not point to a
public server for any data that are not synthetic.

Test status as of 8 October 2026:

- Assessment Bundles and claim Bundles pass the HL7 FHIR Validator (R4 with the AI
  Transparency implementation guide) with 0 errors.
- A FHIR round trip on 300 randomly selected claims preserves every model input, and the
  resulting probabilities are identical.
- A live test against the public test server `hapi.fhir.org/baseR4` with a synthetic claim
  succeeded: the resources were stored, their references were linked, they could be read
  back, and a repeated delivery created no duplicates.
- The SATUSEHAT connector follows the official documentation (token at
  `/oauth2/v1/accesstoken`, data at `/fhir-r4/v1/{Resource}`) and was tested against a local
  mock server. **It has not been tested in the SATUSEHAT sandbox.** Credentials are issued
  by the Ministry of Health only to health facilities, health offices, and electronic
  medical record vendors, upon application.
- The BPJS Kesehatan claim module in SATUSEHAT is itself available only in the sandbox, and
  its public playbook currently ends at `Invoice`. The `Claim` structure used here therefore
  follows the documented mandatory fields, with the INA-CBG code placed in
  `diagnosis.packageCode`.

Contents of the assessment Bundle:

- `RiskAssessment`: the probability of each severity level, with the security label `AIAST`.
- `Provenance`: NADI as author, the verifier as verifier once a decision exists, and the
  claim as source.
- `Device`: the identity of the model. The version is taken from the hash of the checkpoint
  file.

Provenance and Device follow the AI-Provenance and AI-Device profiles of the HL7
implementation guide [AI Transparency on FHIR](https://build.fhir.org/ig/HL7/aitransparency-ig/)
(STU1 ballot, based on R4).

As of 5 October 2026, the SATUSEHAT documentation lists `RiskAssessment`, `Claim`, and
`ClaimResponse`, but not `Provenance` or `Device`. The score can therefore be carried by
SATUSEHAT, whereas the record of which system produced it and who verified it cannot yet.

Identifiers in the Bundles are placeholders. In production they would be the SEP number and
the IHS patient identifier. The code system URLs for INA-CBG and hospital class are also
placeholders.

## Getting started

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# regular sets of the 2021 to 2025 editions, then train on claims up to 2023
.venv/bin/python prepare.py <edition>/data <edition>/data ... --out data/claims_all.parquet
.venv/bin/python train.py --data data/claims_all.parquet --before 2024-01-01 --steps 10000   # 52 minutes
.venv/bin/python validate.py    # test on 2024; --rules-only skips the model
# the application reads the 2025 edition
.venv/bin/python prepare.py "/path/to/Data Sampel Reguler Edisi 2025/data"
.venv/bin/python evaluate.py    # about 5 minutes; writes artifacts/scored.parquet
.venv/bin/uvicorn app:app       # then open http://localhost:8000
```

`python test_nadi.py` runs a quick check of the tokenizer, the rules, the facility index, and
the FHIR bridge.


## Repository structure

| File | Contents |
|---|---|
| `prepare.py` | Data cleaning, from `.dta` files to a single table of claims |
| `nadi.py` | Tokenizer, model, scoring, rule-based checks, and facility index |
| `train.py` | Training |
| `evaluate.py` | Evaluation with injected anomalies, and scores for the full sample |
| `validate.py` | Test on a later year, rule hits per year, and facility recurrence across years |
| `bridge.py` | Claims and assessments as FHIR R4 in both directions, and the SATUSEHAT client |
| `app.py`, `web/` | Priority queue, facility profile, claim simulator, and reviewer decisions |
| `docs/`, `make_demo.py` | Public project page (GitHub Pages), precomputed demo answers for synthetic claims, and the overview video |
| `make_app_demo.py`, `web/static.js`, `docs/app/` | The application on synthetic claims as static files: answers are computed ahead of time, so neither the data nor the model is published |
| `MODEL_CARD.md` | Model card for NADI (in Indonesian) |

## Data and privacy

The BPJS Kesehatan sample data, the trained model, and all files derived from the data are
not included in this repository. Member and facility identifiers are reshuffled during data
cleaning, and the data contain no national identity numbers or names. The claims in the
public demonstration are synthetic.
