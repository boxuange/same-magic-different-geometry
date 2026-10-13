# Same Magic, Different Geometry

**Minimal verification code and computer-assisted proof data for the paper *Same Magic, Different Geometry*.**

This repository provides executable checks for two finite six-tensor holographic quantum codes. It covers four research stages (S39, S50, S56 and S59), including the specified leading-order resource-matching certificates and exact microscopic Pauli calculations for regional proto-area witnesses.

This is a **minimal verification release**, not the full research history or a claim that every upstream calculation is regenerated. In particular, the S39 and S56 fixed-point checks use archived physical Gram-matrix error bounds as inputs; generating those bounds directly from the full physical tensor contractions is outside the scope of this release.

## Run the verification

Python 3.11–3.13 is recommended; the package has been tested with Python 3.13 and NumPy 2.3.5. After installing the dependencies, no GPU, credentials or internet access are required.

From the repository root:

```bash
python -m pip install -r requirements.txt
python verify_all.py
```

A successful run ends with:

```text
ALL PASS — 7 command groups, including 4 S59 sub-checks.
LIMIT: S39/S56 physical-error inputs are audited archive certificates, not reconstructed from full exact tensor orbit.
```

The runner writes `verification_report.json`, which records the per-command exit codes and output excerpts. Other generated JSON reports are listed in `.gitignore`; they are created automatically by the verification scripts. There is **no need to download or prepopulate them**.

You can also verify the fixed distributed files before or after running the tests:

```bash
python check_integrity.py
```

This checks the SHA-256 manifest of the distributed **source code, frozen inputs and archived certificate inputs**. Generated output reports are not part of the fixed manifest. The checker accepts platform-specific text line endings when validating files; this does not affect the binary `.npy` and `.npz` inputs.

## What is checked?

| Stage | Rerun checks | Limitation |
| --- | --- | --- |
| `S39/` | Exact rational fixed-point inequalities and bridge witness bound, using archived eight-resource Gram error enclosures | Does not regenerate physical Gram error enclosures from the full original tensor network |
| `S50/` | Local Pauli Gram calculations, rank-eight structure, dyadic frozen-input check and 256 integer tests for `235 -> 35` erasure | Does not regenerate S39's eight physical resource Gram constructions |
| `S56/` | Frozen-input fingerprints, arithmetic consistency of nine archived square-network physical-error bounds, and exact rational fixed-point contraction inequalities | Does not regenerate the nine upstream physical Gram certificates |
| `S59/` | 128 real-basis source-transfer checks; 256 + 64 integer erasure checks (`235 -> 25 -> 5`); 195 Clifford–Pauli routes; exact rank-two witness and sharpened lower bound | Strict existence of the matched-resource control uses the archived S56 certificate |

The top-level runner `verify_all.py` invokes seven command groups, including the nested four-check S59 suite. In particular, S59's dyadic Pauli coefficients are regenerated first and then used to check its Clifford Gram and strict witness bound.

### Main entry points

- `S39/verify_full_frozen_code_s39.py`: bridge resource-equivalence fixed-point verification, conditional on archived physical-error bounds.
- `S50/s50_exact_local_pauli_gram.py`: exact rank-eight local bridge Pauli Gram from the reference integer/dyadic prescription.
- `S50/s50_exact_binary64_physical.py`: bridge Pauli Gram calculation from the frozen physical binary64 input; supports the stricter bridge bound `>1.0321e-5` with the certified S39 pivot ball.
- `S50/s50_exact_pauli_erasure.py`: integer bridge `235 -> 35` Pauli-erasure identities.
- `S56/verify_s56_certified_chart.py`: square physical-error record checks and fixed-point contraction certificate (pivot radius `1e-5`).
- `S59/S58/square_pauli_exact_dyadic.py`: recomputes local dyadic Pauli coefficients needed by S59.
- `S59/run_s59.py`: runs exact source transfer, nested erasures, Clifford Gram, and sharpened witness checks.

The second-topology square witness lower bound is `>2.4209e-6` **given the S56 certified resource-equivalent pivot chart**. The code also checks the exact rank-two operator mechanism for the specified frozen input and orbit.

## Exact inputs and proof scope

The release includes the bridge and square frozen binary64 Hermitian arrays (`S50/bridge_weighted_H_binary64.npy` and `S56/s55_frozen_binary64_weighted_H.npy`), corresponding Gram and certificate records, and exact stabiliser/Clifford calculation code. In the mathematical statements, each binary64 entry is interpreted as an exact dyadic rational. The micro-operator calculations rerun integer or rational identities; numerical comparisons are labelled separately.

**An important distinction:** S39 and S56 test exact rational consequences of supplied physical-error certificates. They do **not** independently rederive every physical Gram error enclosure from its raw six-tensor contraction. Accordingly, `PASS` means the released checks succeeded, not that this repository supplies an end-to-end independent reproof of every upstream physical bound.

These calculations concern fixed finite encodings, selected leading-order resource coefficients, prescribed graph-local response models, and smooth source-independent near-decoder **unitary** recovery conventions. They make no claim about arbitrary CPTP-optimal recovery, equality of all resources at finite deformation strength, a universal geometry criterion, or continuum gravitational dynamics.

For full definitions, proof hypotheses and geometric-witness conventions, consult the accompanying paper and Supplemental Material.

## File integrity

`SHA256_MANIFEST.json` lists the *fixed* distributed files, excluding itself. Run `python check_integrity.py` to verify it. The SHA-256 hashes certify file identity, not mathematical correctness. Python-generated outputs are intentionally excluded from the manifest and ignored by Git, so the manifest still verifies after running `verify_all.py`.

## Citation and reuse

Please cite *Same Magic, Different Geometry* when using these verification materials. Paper and version-specific archive DOIs can be added after assignment. **No source-code licence has been granted in this release**; the repository owner may choose one separately if code redistribution or derivative reuse is intended.
