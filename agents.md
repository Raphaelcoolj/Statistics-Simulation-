# Agent References

## How Example Scenarios Work

When predictions are generated, example scenarios show predicted values. The system uses a two-tier approach:

1. **Primary**: Uses actual data rows from `schema.sampleRows` (first 200 rows, max 10 for column stats)
2. **Fallback**: If sample rows lack valid numeric values for all predictors, generates synthetic examples using column statistics (median/mean/min-max midpoint)

### Satisfaction Rating Prediction Example

When Efficacy_Rate_Pct is 83.0, Adverse_Events_Pct is 12.9, and Side_Effects_Count is 4.0 (plus 1 additional factor), the model predicts a Satisfaction_Rating of 40.26. This example illustrates how input factors influence the predicted satisfaction rating.

## Key Files

- `lib/stats/parser.ts:33` - Sample rows limited to 10 for performance
- `app/analyse/page.tsx:903-970` - Example prediction generation logic
- `lib/types.ts:1-10` - Column interface with mean/median fields