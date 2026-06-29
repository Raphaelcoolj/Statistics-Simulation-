import { z } from 'zod'

const modelTypeEnum = z.enum(['linear', 'polynomial', 'logistic', 'multiple', 'timeseries', 'randomforest'])

export const columnSchema = z.object({
  name: z.string().min(1),
  type: z.enum(['continuous', 'categorical', 'ordinal', 'datetime', 'binary']),
  uniqueValues: z.array(z.union([z.string(), z.number()])).optional(),
  min: z.number().optional(),
  max: z.number().optional(),
  sampleValues: z.array(z.unknown()).optional(),
  nullCount: z.number().optional(),
})

export const datasetSchemaSchema = z.object({
  fileName: z.string().min(1),
  rowCount: z.number().nonnegative(),
  columnCount: z.number().nonnegative(),
  columns: z.array(columnSchema).min(1),
  sampleRows: z.array(z.record(z.string(), z.unknown())),
})

export const hypothesisTestSchema = z.object({
  type: z.enum(['t-test', 'chi-square', 'anova']),
  columns: z.array(z.string()).min(1),
})

export const analysisRequestSchema = z.object({
  mode: z.enum(['smart', 'manual']),
  descriptive: z.object({
    columns: z.array(z.string()).min(1),
    measures: z.array(z.enum(['central', 'spread', 'distribution'])),
  }).optional(),
  inferential: z.object({
    correlationPairs: z.array(z.tuple([z.string(), z.string()])).optional(),
    hypothesisTests: z.array(hypothesisTestSchema).optional(),
    regression: z.object({
      dependent: z.string().min(1),
      predictors: z.array(z.string()).min(1),
    }).optional(),
  }).optional(),
  predictive: z.object({
    dependent: z.string().min(1),
    predictors: z.array(z.string()).min(1),
    modelType: modelTypeEnum.optional(),
  }).optional(),
})

export const profileRequestBodySchema = z.object({
  schema: datasetSchemaSchema,
})

export const analyseRequestBodySchema = z.object({
  schema: datasetSchemaSchema.optional(),
  analyses: analysisRequestSchema,
})

export const interpretRequestBodySchema = z.object({
  schema: datasetSchemaSchema,
  result: z.object({
    descriptive: z.array(z.any()).optional(),
    inferential: z.any().optional(),
    predictive: z.any().optional(),
    chartSuggestions: z.array(z.any()).optional(),
  }),
})
