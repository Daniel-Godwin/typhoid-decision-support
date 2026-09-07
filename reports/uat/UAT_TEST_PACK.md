# User Acceptance Test Pack

**System:** Typhoid Diagnostic Decision Support v2.0.0  
**URL:** https://typhoid-decision-support.onrender.com  
**Model under test:** `svm_binary.joblib` (SHA-256 prefix `13b5620b5e898412`)  
**Referral threshold:** 0.08  
**Generated:** 07 September 2026 by `python scripts/make_uat_samples.py`

## What this pack is for

Ten patient records drawn from the held-out test partition — data the model never saw during training. Each is entered into the deployed system through the normal clinician workflow, and the result compared against the **expected system output** stated below.

The expected outputs were produced by running the same saved model artefact that the deployed service loads. They are what the system should return, not a guess.

## Acceptance criteria

The test **passes** when, for all ten cases:

1. The record is accepted without a validation error.
2. The prediction matches the **Expected result** column.
3. A calibrated probability and a triage recommendation are shown.
4. The contributing findings are listed.
5. The assessment is still present after signing out and back in.

The criterion is agreement with the **expected system output**, not with the patient's recorded diagnosis. Two of the ten differ, and that is documented below rather than discovered during the test.

## Declared limitation — please read before testing

Cases **4, 8** are typhoid patients whom the system will report as **No Typhoid**. This is a known and documented limitation, not a fault in the deployment.

In the source dataset every patient with a fever duration of one day or more is labelled typhoid, without a single exception in 8,981 records. The model has learned that rule. These two patients were recorded with a fever duration of zero, so the rule cannot reach them — and neither can the model, which is why they are in this pack. Excluding them would make the system look better than it is.

Case **5** is the counterpart: a typhoid patient also recorded with a fever duration of zero, whom the system does identify correctly. It is included so the pack is not weighted towards failure.

One further observation makes the point sharply. Every one of the 1,799 typhoid patients in the test partition with a fever duration of one day or more receives a probability of at least 0.99998 — not one falls below 0.999. The model expresses no uncertainty at all about any case the rule covers. The only patients it is genuinely unsure about are the zero-fever cases, which is where a real diagnostic judgement would actually be required.

The full analysis is in `docs/DATASET_AUDIT.md` and `reports/THRESHOLD_ANALYSIS.md`. The short version: the reported 98.8% accuracy measures how the dataset was constructed, not diagnostic ability, and the system must not be used on real patients on this training data.

## Summary of expected results

| Case | Patient | Recorded diagnosis | Expected result | Probability | Type |
|---|---|---|---|---:|---|
| 1 | UAT-001 | Typhoid | **Typhoid** | 1.0000 | routine |
| 2 | UAT-002 | No Typhoid | **No Typhoid** | 0.0052 | routine |
| 3 | UAT-003 | No Typhoid | **No Typhoid** | 0.0118 | routine |
| 4 | UAT-004 | Typhoid | **No Typhoid** ⚠ | 0.0053 | known-difficult |
| 5 | UAT-005 | Typhoid | **Typhoid** | 0.9996 | difficult, correctly identified |
| 6 | UAT-006 | No Typhoid | **No Typhoid** | 0.0147 | routine |
| 7 | UAT-007 | Typhoid | **Typhoid** | 1.0000 | routine |
| 8 | UAT-008 | Typhoid | **No Typhoid** ⚠ | 0.0078 | known-difficult |
| 9 | UAT-009 | No Typhoid | **No Typhoid** | 0.0186 | routine |
| 10 | UAT-010 | No Typhoid | **No Typhoid** | 0.0260 | routine |

Agreement with the recorded diagnosis: **8 of 10**. The two disagreements are the declared cases above.

---

## How to run each case

1. Sign in as a clinician.
2. **Clinical → Patients → Register patient.** Enter the patient code, name, age, gender and location from the case below.
3. **Clinical → New assessment.** Select that patient, then enter every clinical field exactly as listed.
4. Click **Run assessment** and record what the system returns.

Three fields — Gastrointestinal Symptoms, Neurological Symptoms and Ongoing Infection in Society — are optional, because roughly a quarter of real records lack them. Where a case says *Not recorded*, leave the field at that setting rather than inventing a value.

---

## The ten cases

### Case 1 — UAT-001

**Patient record**

| Field | Value |
|---|---|
| Patient code | `UAT-001` |
| Full name | Amara Okafor |
| Age | 24 |
| Gender | Male |
| Location | Endemic |

**Assessment fields**

| Field | Value |
|---|---|
| Fever Duration (Days) | 9 |
| White Blood Cell Count | 7147 |
| Platelet Count | 102421 |
| Socioeconomic Status | Low |
| Water Source Type | River |
| Sanitation Facilities | Open Defecation |
| Hand Hygiene | Yes |
| Consumption of Street Food | No |
| Gastrointestinal Symptoms | Abdominal Pain |
| Neurological Symptoms | Delirium |
| Skin Manifestations | No |
| Typhoid Vaccination Status | Not Received |
| Previous History of Typhoid | Yes |
| Weather Condition | Cold & Humid |
| Ongoing Infection in Society | *Not recorded* |
| Widal Test | High O & H Antibody |
| Typhidot Test | IgG Positive |

**Expected result:** Typhoid (probability 1.0000)

**Tester records:**

| | |
|---|---|
| Result returned | |
| Probability shown | |
| Triage recommendation | |
| Matches expected? | Yes / No |
| Notes | |

### Case 2 — UAT-002

**Patient record**

| Field | Value |
|---|---|
| Patient code | `UAT-002` |
| Full name | Bello Yusuf |
| Age | 80 |
| Gender | Female |
| Location | Rural |

**Assessment fields**

| Field | Value |
|---|---|
| Fever Duration (Days) | 0 |
| White Blood Cell Count | 8768 |
| Platelet Count | 402345 |
| Socioeconomic Status | High |
| Water Source Type | Tap |
| Sanitation Facilities | Open Defecation |
| Hand Hygiene | No |
| Consumption of Street Food | No |
| Gastrointestinal Symptoms | Diarrhea |
| Neurological Symptoms | Confusion |
| Skin Manifestations | Yes |
| Typhoid Vaccination Status | Received |
| Previous History of Typhoid | No |
| Weather Condition | Rainy & Wet |
| Ongoing Infection in Society | *Not recorded* |
| Widal Test | Low O & H Antibody |
| Typhidot Test | IgG Positive |

**Expected result:** No Typhoid (probability 0.0052)

**Tester records:**

| | |
|---|---|
| Result returned | |
| Probability shown | |
| Triage recommendation | |
| Matches expected? | Yes / No |
| Notes | |

### Case 3 — UAT-003

**Patient record**

| Field | Value |
|---|---|
| Patient code | `UAT-003` |
| Full name | Chidinma Nwosu |
| Age | 50 |
| Gender | Female |
| Location | Urban |

**Assessment fields**

| Field | Value |
|---|---|
| Fever Duration (Days) | 0 |
| White Blood Cell Count | 7085 |
| Platelet Count | 99319 |
| Socioeconomic Status | Middle |
| Water Source Type | Tap |
| Sanitation Facilities | Proper |
| Hand Hygiene | Yes |
| Consumption of Street Food | Yes |
| Gastrointestinal Symptoms | *Not recorded* |
| Neurological Symptoms | *Not recorded* |
| Skin Manifestations | No |
| Typhoid Vaccination Status | Received |
| Previous History of Typhoid | No |
| Weather Condition | Moderate |
| Ongoing Infection in Society | Dengue Outbreak |
| Widal Test | High O & H Antibody |
| Typhidot Test | IgG Positive |

**Expected result:** No Typhoid (probability 0.0118)

**Tester records:**

| | |
|---|---|
| Result returned | |
| Probability shown | |
| Triage recommendation | |
| Matches expected? | Yes / No |
| Notes | |

### Case 4 — UAT-004

**Patient record**

| Field | Value |
|---|---|
| Patient code | `UAT-004` |
| Full name | Danladi Sule |
| Age | 21 |
| Gender | Male |
| Location | Endemic |

**Assessment fields**

| Field | Value |
|---|---|
| Fever Duration (Days) | 0 |
| White Blood Cell Count | 7721 |
| Platelet Count | 389591 |
| Socioeconomic Status | Low |
| Water Source Type | Untreated Supply |
| Sanitation Facilities | Proper |
| Hand Hygiene | Yes |
| Consumption of Street Food | Yes |
| Gastrointestinal Symptoms | Abdominal Pain |
| Neurological Symptoms | Confusion |
| Skin Manifestations | Yes |
| Typhoid Vaccination Status | Not Received |
| Previous History of Typhoid | No |
| Weather Condition | Moderate |
| Ongoing Infection in Society | Dengue Outbreak |
| Widal Test | High O & H Antibody |
| Typhidot Test | IgM Positive |

**Expected result:** No Typhoid (probability 0.0053)

> This patient's recorded diagnosis is **Typhoid**. The system is expected to disagree — see the declared limitation above. Record this as **pass** if the system returns the expected result.

**Tester records:**

| | |
|---|---|
| Result returned | |
| Probability shown | |
| Triage recommendation | |
| Matches expected? | Yes / No |
| Notes | |

### Case 5 — UAT-005

**Patient record**

| Field | Value |
|---|---|
| Patient code | `UAT-005` |
| Full name | Esther Adeniyi |
| Age | 37 |
| Gender | Female |
| Location | Urban |

**Assessment fields**

| Field | Value |
|---|---|
| Fever Duration (Days) | 0 |
| White Blood Cell Count | 19294 |
| Platelet Count | 373031 |
| Socioeconomic Status | Middle |
| Water Source Type | Untreated Supply |
| Sanitation Facilities | Proper |
| Hand Hygiene | No |
| Consumption of Street Food | Yes |
| Gastrointestinal Symptoms | *Not recorded* |
| Neurological Symptoms | Delirium |
| Skin Manifestations | Yes |
| Typhoid Vaccination Status | Not Received |
| Previous History of Typhoid | Yes |
| Weather Condition | Rainy & Wet |
| Ongoing Infection in Society | Seasonal Flu |
| Widal Test | Low O & H Antibody |
| Typhidot Test | IgM Positive |

**Expected result:** Typhoid (probability 0.9996)

**Tester records:**

| | |
|---|---|
| Result returned | |
| Probability shown | |
| Triage recommendation | |
| Matches expected? | Yes / No |
| Notes | |

### Case 6 — UAT-006

**Patient record**

| Field | Value |
|---|---|
| Patient code | `UAT-006` |
| Full name | Farouk Bala |
| Age | 79 |
| Gender | Female |
| Location | Urban |

**Assessment fields**

| Field | Value |
|---|---|
| Fever Duration (Days) | 0 |
| White Blood Cell Count | 6154 |
| Platelet Count | 209829 |
| Socioeconomic Status | Low |
| Water Source Type | River |
| Sanitation Facilities | Open Defecation |
| Hand Hygiene | No |
| Consumption of Street Food | Yes |
| Gastrointestinal Symptoms | Abdominal Pain |
| Neurological Symptoms | Headache |
| Skin Manifestations | No |
| Typhoid Vaccination Status | Not Received |
| Previous History of Typhoid | No |
| Weather Condition | Rainy & Wet |
| Ongoing Infection in Society | COVID-19 Surge |
| Widal Test | High O & H Antibody |
| Typhidot Test | IgM Positive |

**Expected result:** No Typhoid (probability 0.0147)

**Tester records:**

| | |
|---|---|
| Result returned | |
| Probability shown | |
| Triage recommendation | |
| Matches expected? | Yes / No |
| Notes | |

### Case 7 — UAT-007

**Patient record**

| Field | Value |
|---|---|
| Patient code | `UAT-007` |
| Full name | Grace Mensah |
| Age | 15 |
| Gender | Female |
| Location | Urban |

**Assessment fields**

| Field | Value |
|---|---|
| Fever Duration (Days) | 21 |
| White Blood Cell Count | 7256 |
| Platelet Count | 262741 |
| Socioeconomic Status | Middle |
| Water Source Type | River |
| Sanitation Facilities | Proper |
| Hand Hygiene | Yes |
| Consumption of Street Food | No |
| Gastrointestinal Symptoms | *Not recorded* |
| Neurological Symptoms | *Not recorded* |
| Skin Manifestations | No |
| Typhoid Vaccination Status | Received |
| Previous History of Typhoid | Yes |
| Weather Condition | Hot & Dry |
| Ongoing Infection in Society | *Not recorded* |
| Widal Test | High O & H Antibody |
| Typhidot Test | IgG Positive |

**Expected result:** Typhoid (probability 1.0000)

**Tester records:**

| | |
|---|---|
| Result returned | |
| Probability shown | |
| Triage recommendation | |
| Matches expected? | Yes / No |
| Notes | |

### Case 8 — UAT-008

**Patient record**

| Field | Value |
|---|---|
| Patient code | `UAT-008` |
| Full name | Hauwa Ibrahim |
| Age | 16 |
| Gender | Female |
| Location | Rural |

**Assessment fields**

| Field | Value |
|---|---|
| Fever Duration (Days) | 0 |
| White Blood Cell Count | 6129 |
| Platelet Count | 444779 |
| Socioeconomic Status | Low |
| Water Source Type | River |
| Sanitation Facilities | Open Defecation |
| Hand Hygiene | Yes |
| Consumption of Street Food | No |
| Gastrointestinal Symptoms | Abdominal Pain |
| Neurological Symptoms | Delirium |
| Skin Manifestations | No |
| Typhoid Vaccination Status | Received |
| Previous History of Typhoid | Yes |
| Weather Condition | Rainy & Wet |
| Ongoing Infection in Society | Dengue Outbreak |
| Widal Test | High O & H Antibody |
| Typhidot Test | IgG Positive |

**Expected result:** No Typhoid (probability 0.0078)

> This patient's recorded diagnosis is **Typhoid**. The system is expected to disagree — see the declared limitation above. Record this as **pass** if the system returns the expected result.

**Tester records:**

| | |
|---|---|
| Result returned | |
| Probability shown | |
| Triage recommendation | |
| Matches expected? | Yes / No |
| Notes | |

### Case 9 — UAT-009

**Patient record**

| Field | Value |
|---|---|
| Patient code | `UAT-009` |
| Full name | Ikenna Obi |
| Age | 59 |
| Gender | Female |
| Location | Endemic |

**Assessment fields**

| Field | Value |
|---|---|
| Fever Duration (Days) | 0 |
| White Blood Cell Count | 4837 |
| Platelet Count | 172084 |
| Socioeconomic Status | Low |
| Water Source Type | Untreated Supply |
| Sanitation Facilities | Open Defecation |
| Hand Hygiene | Yes |
| Consumption of Street Food | Yes |
| Gastrointestinal Symptoms | Diarrhea |
| Neurological Symptoms | Headache |
| Skin Manifestations | No |
| Typhoid Vaccination Status | Received |
| Previous History of Typhoid | Yes |
| Weather Condition | Moderate |
| Ongoing Infection in Society | Dengue Outbreak |
| Widal Test | High O & H Antibody |
| Typhidot Test | IgM Positive |

**Expected result:** No Typhoid (probability 0.0186)

**Tester records:**

| | |
|---|---|
| Result returned | |
| Probability shown | |
| Triage recommendation | |
| Matches expected? | Yes / No |
| Notes | |

### Case 10 — UAT-010

**Patient record**

| Field | Value |
|---|---|
| Patient code | `UAT-010` |
| Full name | Jamila Aliyu |
| Age | 77 |
| Gender | Female |
| Location | Endemic |

**Assessment fields**

| Field | Value |
|---|---|
| Fever Duration (Days) | 0 |
| White Blood Cell Count | 3450 |
| Platelet Count | 56336 |
| Socioeconomic Status | High |
| Water Source Type | River |
| Sanitation Facilities | Proper |
| Hand Hygiene | Yes |
| Consumption of Street Food | No |
| Gastrointestinal Symptoms | Constipation |
| Neurological Symptoms | *Not recorded* |
| Skin Manifestations | Yes |
| Typhoid Vaccination Status | Received |
| Previous History of Typhoid | No |
| Weather Condition | Moderate |
| Ongoing Infection in Society | COVID-19 Surge |
| Widal Test | Low O & H Antibody |
| Typhidot Test | Negative |

**Expected result:** No Typhoid (probability 0.0260)

**Tester records:**

| | |
|---|---|
| Result returned | |
| Probability shown | |
| Triage recommendation | |
| Matches expected? | Yes / No |
| Notes | |

---

## Sign-off

| | |
|---|---|
| Cases passed | ___ / 10 |
| Tester name | |
| Date | |
| Overall result | Accepted / Accepted with comments / Rejected |
| Comments | |
