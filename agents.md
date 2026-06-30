# Agent References

## How Example Scenarios Work

When predictions are generated, example scenarios show predicted values. The system uses a two-tier approach:

1. **Primary**: Uses actual data rows from `schema.sampleRows` (all rows passed through; example prediction filters for numeric validity)
2. **Fallback**: If sample rows lack valid numeric values for all predictors, generates synthetic examples using column statistics (median/mean/min-max midpoint)

### Satisfaction Rating Prediction Example

When Efficacy_Rate_Pct is 83.0, Adverse_Events_Pct is 12.9, and Side_Effects_Count is 4.0 (plus 1 additional factor), the model predicts a Satisfaction_Rating of 40.26. This example illustrates how input factors influence the predicted satisfaction rating.

## Key Files

- `lib/stats/parser.ts:33` - Sample rows set to all data (no limit) for chart rendering
- `lib/stats/predictive.ts:585-586` - Train/test split called once (fixed bug: was called twice, producing invalid test metrics)
- `lib/stats/predictive.ts:228` - Ridge regression fallback for singular matrices (was: fell back to 1-predictor linear)
- `lib/stats/predictive.ts:306` - Logistic: 5000 steps, L2=0.01, retry with 10000 steps + L2=0.1 (was: 1000 steps, no reg)
- `lib/stats/preprocessing.ts:121` - Imputation uses all rows before dropping dependent-missing (was: dropped first, imputed on subset)
- `app/analyse/page.tsx:456` - Scatter chart samples max 2000 points from full dataset (was: used 10 sample rows, no sampling)
- `app/analyse/page.tsx:1014-1022` - Model failed state shown when predictive model errors out
- `app/analyse/page.tsx:593-597` - Chart fallback shows "Chart not available" message instead of row-indexed labels
- `lib/types.ts:1-10` - Column interface with mean/median fields

## Build & Test

```bash
npm run build        # next build (includes typecheck + lint)
npm test             # vitest
npx tsc --noEmit     # typecheck only
```