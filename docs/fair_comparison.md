# DIPY vs ANTs SyN: Fair Comparison Guide

*Last reviewed: 2026-08-26*

## Purpose

This document summarizes the current working understanding of the parameter correspondences and implementation differences between DIPY's `SymmetricDiffeomorphicRegistration` and ANTs/ANTsPy SyN registration. It is intended as a guide for designing fair benchmarks and for discussing remaining uncertainties.

The focus here is **SyN-only deformable registration**, not affine, rigid, or full multi-stage registration pipelines.

---

## Scope and Working Assumptions

DIPY and ANTs expose SyN registration through substantially different interfaces:

* **DIPY** uses a dedicated class for SyN-like registration: `SymmetricDiffeomorphicRegistration`.
* **ANTsPy** uses a single high-level `registration()` function whose behavior depends on `type_of_transform`.

For fair SyN-only comparisons, the closest ANTs mode is generally:

```python
ants.registration(..., type_of_transform="SyNOnly")
```

rather than:

```python
ants.registration(..., type_of_transform="SyN")
```

because ANTsPy `"SyN"` includes an affine stage before the deformable SyN stage, whereas DIPY's `SymmetricDiffeomorphicRegistration` only performs the nonlinear symmetric diffeomorphic optimization.

---

## Parameter Correspondences and Match Quality

The closest-looking arguments do not always produce identical numerical
behavior. The table therefore distinguishes direct role matches from merely
conceptual correspondences.

| Purpose | DIPY | ANTs / ANTsPy | Match quality and caveats |
| ------- | ----- | ------------- | ------------------------- |
| Nonlinear iterations by pyramid level | `level_iters` | `reg_iterations` | Close schedule mapping. Both sequences specify the maximum iterations per level, but pyramid construction and stopping behavior still differ. |
| Update magnitude | `step_length` | `grad_step` | Same role, not an exact numerical equivalence: normalization and field composition differ. |
| Reference image | `static` in `optimize()` | `fixed` | Direct data-role match. |
| Image being registered | `moving` in `optimize()` | `moving` | Direct data-role match. |
| Initial alignment | `prealign` | `initial_transform` | Same role but different representations. This benchmark uses no additional affine transform in DIPY and `"Identity"` in ANTsPy after creating shared prealigned inputs. |
| CC neighborhood size | `CCMetric(radius=...)` | `syn_sampling` with `syn_metric="CC"` | Close argument mapping: `syn_sampling` is the local CC radius in this mode. The CC implementations are not expected to return identical energies or derivatives. |
| MI histogram resolution | `MIMetric(nbins=...)` | `syn_sampling` with `syn_metric="mattes"` | Close argument mapping: `syn_sampling` is the number of histogram bins in this mode. Matching the bin count does not make the histogram estimators or derivatives identical. |
| Convergence threshold | `opt_tol` | `1e-7` emitted by ANTsPy for the SyN stage | Same role, but the convergence calculations differ. The default benchmark config sets DIPY to `1e-7`; ANTsPy does not expose this value as a `registration()` argument. |
| Convergence window | `energy_window` | Window size `8` emitted by ANTsPy | Same role, but the monitored quantities and stopping calculations differ. DIPY defaults to `12`; this benchmark explicitly sets it to `8`. |


---

## Similarity Metrics: CC and MI

The benchmark supports the same two metric families for each backend. Matching
the family and its main sampling parameter is necessary for comparison, but it
does not make the implementations mathematically or numerically identical.
The shared `registration.metric` configuration field selects the metric family
for both backends.

### Cross-correlation (CC)

* DIPY uses `CCMetric(dim=3, radius=..., sigma_diff=...)`.
* ANTsPy uses `syn_metric="CC"` and interprets `syn_sampling` as the local
  neighborhood radius.
* The benchmark maps `cc_radius` to both `CCMetric.radius` and `syn_sampling`.
* DIPY's `sigma_diff` and ANTs' `flow_sigma` both regularize update fields, but
  they are only conceptual counterparts because they are applied inside
  different metric and optimizer implementations.

### Mutual information (MI)

* DIPY SyN uses `MIMetric(dim=3, nbins=..., smooth=...)` from
  `dipy.align.metrics`. This is the deformable-registration metric used by
  `SymmetricDiffeomorphicRegistration`; it should not be confused with DIPY's
  affine-registration `MutualInformationMetric`.
* ANTsPy uses the corresponding metric family through `syn_metric="mattes"`
  and interprets `syn_sampling` as the number of histogram bins.
* The benchmark maps `mi_nbins` to both `MIMetric.nbins` and `syn_sampling`;
  the current default is 32 bins for each backend.
* DIPY's `MIMetric.smooth` and ANTs' `flow_sigma` both regularize update fields,
  but this is again a conceptual correspondence rather than a numerical
  equivalence. DIPY has no counterpart to ANTs' `total_sigma`.

In particular, `syn_sampling` is overloaded by ANTsPy: it means radius for CC
and histogram-bin count for Mattes MI. It must therefore always be interpreted
together with `syn_metric`.

---

## Pyramid Construction: Smoothing and Scaling

### ANTs / ANTsPy

For the SyN stage, ANTsPy computes scalar smoothing sigmas and nominal shrink factors from the number of pyramid levels:

```text
iterations_i      = reg_iterations[i]
smoothing_sigma_i = L - 1 - i
shrink_factor_i   = 2 ** (L - 1 - i)
```

Example:

```text
reg_iterations = [40, 20, 20]
L = 3
smoothing sigmas = [2, 1, 0]
shrink factors   = [4, 2, 1]
```

Later, ANTs converts the nominal shrink factor into per-dimension integer shrink factors, taking image spacing into account. The goal is to keep the downsampled image spacing as close as possible to isotropic while using integer shrink factors.

### DIPY

DIPY constructs a `ScaleSpace` object. For each pyramid level, it starts from the same conceptual scale progression:

```text
scale_i = 2 ** i
```

but then computes per-dimension scaling using the minimum input spacing:

```text
scaling[d] = 2**i * min_spacing / input_spacing[d]
```

This gives:

```text
output_spacing[d] = input_spacing[d] * scaling[d]
                  = 2**i * min_spacing
```

Therefore, DIPY allows non-integer per-dimension scaling values and effectively forces the target spacing at each level to be isotropic in physical units.

For smoothing, DIPY computes voxel-unit Gaussian sigmas as:

```text
sigma[d] = sigma_factor * (output_spacing[d] / input_spacing[d] - 1)
```

Thus, DIPY's default smoothing can be much weaker than ANTs' default smoothing. For three isotropic levels and `sigma_factor = 0.2`, DIPY gives fine-to-coarse sigmas:

```text
[0, 0.2, 0.6]
```

whereas ANTs uses:

```text
[0, 1, 2]
```

fine-to-coarse. For Affine Registration, DIPY does accept the smoothing sigmas as a list, allowing for using the same smoothing scheme as ANTs for example.

---

## Update Smoothing and Deformation Regularization

### ANTs

ANTs separates deformation regularization into two explicit SyN parameters:

```text
SyN[grad_step, flow_sigma, total_sigma]
```

* `flow_sigma` smooths the **update field**, i.e. the incremental displacement estimated at the current iteration before it is composed into the running deformation.
* `total_sigma` smooths the **accumulated total field**, i.e. the running deformation after update composition.

### DIPY

DIPY's `SymmetricDiffeomorphicRegistration` does not expose direct equivalents of either `flow_sigma` or `total_sigma`.

Instead, update smoothing is implemented inside metric classes, before the update field is normalized and composed by the SyN optimizer. For the metrics used here, the relevant arguments are `CCMetric.sigma_diff` and `MIMetric.smooth`. They are therefore at most conceptually similar to ANTs' `flow_sigma`, not `total_sigma`.

There is currently no direct DIPY equivalent of ANTs' `total_sigma`.

---

## Masks

ANTs supports explicit metric masks:

```text
mask
moving_mask
mask_all_stages
```

These can restrict where the metric is evaluated during optimization.

DIPY `SymmetricDiffeomorphicRegistration` does not expose equivalent fixed/moving metric masks. The closest related mechanism is `metric.mask0`, used by `ScaleSpace` to keep zero-valued regions zero after smoothing. This is not equivalent to ANTs' fixed/moving registration masks.

---

## Other ANTs Features Without Direct DIPY Equivalents

| ANTs parameter                  | Relevance to SyN                                           | DIPY equivalent                                              |
| ------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------ |
| `restrict_transformation`       | Can restrict deformation components along selected axes    | No built-in equivalent                                       |
| `multivariate_extras`           | Adds weighted extra metrics during deformable optimization | No built-in equivalent; would require custom combined metric |
| `use_legacy_histogram_matching` | Legacy intensity preprocessing option; not recommended     | No direct equivalent                                         |

---

## Intensity Preprocessing / Normalization

Current working summary:

* ANTs/ITK appears to build a 256-bin histogram, estimate intensity bounds using lower and upper quantiles, and use those bounds for min/max-like normalization before constructing coarser smoothed levels.
* DIPY performs direct min/max normalization to `[0, 1]` in `ScaleSpace`, and does this again after smoothing at each coarse level.

---

## Inverse Field Computation

Both DIPY and ANTs/ITK SyN maintain inverse displacement fields during optimization.

### DIPY

DIPY exposes inverse-field inversion parameters directly:

```text
inv_iter = 20
inv_tol  = 1e-3
```

These are passed to the displacement-field inversion routine during optimization.

### ANTs / ITK SyN

ANTs/ITK SyN also uses iterative inverse displacement field estimation. The apparent internal settings are:

```text
maximum inverse iterations = 20
mean error tolerance       = 0.001
max error tolerance        = 0.1
```

ANTsPy `registration()` does not expose these inverse-field inversion parameters.

---

## Timing Boundary

The primary timing fields, `dipy_syn_sec` and `ants_syn_sec`, measure the
closest practical public-interface boundary: execution starts after each
backend's fixed and moving intensity images have been loaded and stops when the
warped moving-intensity image exists in memory.

For DIPY, the timed region includes:

1. constructing `CCMetric` or `MIMetric`;
2. constructing and configuring `SymmetricDiffeomorphicRegistration`;
3. running `sdr.optimize()` to estimate the deformation mapping; and
4. running `mapping.transform(moving)` to produce the final warped intensity
   image.

For ANTsPy, the timed region is the complete `ants.registration()` call using
`type_of_transform="SyNOnly"`. Its returned `warpedmovout` already exists when
the function returns, so timing only DIPY's `optimize()` call would omit DIPY's
corresponding final interpolation.

Both timings exclude input-image loading, shared preprocessing, rigid
prealignment, label warping, NIfTI writing, and evaluation.
They should be interpreted as **time to an in-memory warped intensity image**,
not as pure optimizer time.

The timing boundaries are not identical because the two backends expose
different public interfaces and may perform slightly different computations
within them. The benchmark uses the available public calls to provide the
fairest practical comparison while acknowledging this remaining limitation.

---

## Recommended Benchmarking Strategy

1. **Start with SyN-only comparisons**

   * ANTs: `type_of_transform="SyNOnly"`
   * DIPY: `SymmetricDiffeomorphicRegistration`

2. **Use the same metric family where possible**

   * Monomodal comparison: ANTs `syn_metric="CC"` vs DIPY `CCMetric`.
   * Multimodal comparison: ANTs `syn_metric="mattes"` vs DIPY `MIMetric`.
   * For CC, map `syn_sampling` to `CCMetric.radius`.
   * For MI, map `syn_sampling` to `MIMetric.nbins`.

3. **Control pyramid levels explicitly**

   * Match `reg_iterations` and `level_iters`.
   * Be aware that smoothing and scaling schedules are still not identical.

4. **Minimize convergence stopping differences initially**

   * Consider forcing full iteration execution by setting very strict or very permissive convergence thresholds, depending on interface possibilities.
   * Revisit convergence once other differences are controlled.

5. **Set ANTs `total_sigma=0` for fairer comparison**

   * DIPY does not expose a corresponding total-field smoothing parameter.

6. **Treat update smoothing carefully**

   * For CC, compare ANTs `flow_sigma` with DIPY `CCMetric.sigma_diff`, but remember this is only conceptual.
   * For MI, compare ANTs `flow_sigma` with DIPY `MIMetric.smooth`, with the same caveat.

7. **Avoid masks, multivariate metrics, and restricted transforms in the first benchmark**

   * These have no direct DIPY equivalents.
