**Adjusted effect of menu properties on the per-image offer F1 (all models, P1, test)**

| Factor | Images | Change in offer F1 (points) | 95% CI |
|---|---|---|---|
| L_VARIANT | 45 | -17.0 | [-26.0, -9.5] |
| L_COMBO_SET | 37 | -12.0 | [-20.2, -4.1] |
| P_INHERITED | 64 | -9.4 | [-14.0, -3.7] |
| P_MISSING | 51 | -8.6 | [-14.4, -3.7] |
| P_MULTI_UNLABELED | 49 | -8.2 | [-14.2, -2.5] |
| N_SECTION_COMPOSE | 77 | -7.8 | [-12.0, -3.6] |
| P_IMPLICIT_K | 35 | -7.2 | [-15.1, +1.5] |
| D_DISTRACTOR | 101 | -4.2 | [-8.6, -0.1] |
| V_HANDWRITING | 42 | -4.1 | [-12.1, +2.2] |
| P_UNIT_BUNDLE | 73 | -2.9 | [-8.1, +1.5] |
| V_GEOMETRY | 23 | -2.2 | [-10.4, +4.5] |
| S_LAYOUT | 86 | -2.2 | [-6.9, +3.6] |
| L_ADDON | 59 | -1.8 | [-6.3, +3.2] |
| L_ITEM_SPLIT | 28 | -1.7 | [-9.1, +5.5] |
| V_DAMAGE | 86 | -0.4 | [-4.7, +4.5] |
| N_NORMALIZE | 167 | +0.2 | [-5.3, +5.4] |
| V_QUALITY | 110 | +0.4 | [-3.5, +4.2] |
| N_FOREIGN | 97 | +1.9 | [-1.8, +7.0] |
| S_DETACHED | 62 | +2.4 | [-2.0, +7.4] |
| log2(offers) | 272 | +2.9 | [+0.5, +4.9] |
| P_FORMAT | 238 | +6.8 | [-0.1, +13.6] |
| P_HANDWRITTEN_OVERRIDE | 21 | +7.7 | [-0.4, +16.6] |

Linear model with one indicator per tag (tags on at least 10 images), log2 of the number of offers and a model effect; 272 images x 32 models; intervals from 300 resamples of images. A tag coefficient is the change when the tag is present, the other tags and the menu size being equal; log2(offers) is the change per doubling.
