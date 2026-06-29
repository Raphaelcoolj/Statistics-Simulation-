export interface Column {
  name: string
  type: "continuous" | "categorical" | "ordinal" | "datetime" | "binary"
  uniqueValues?: string[] | number[]
  min?: number
  max?: number
  sampleValues?: unknown[]
  nullCount?: number
}

export interface DatasetSchema {
  fileName: string
  rowCount: number
  columnCount: number
  columns: Column[]
  sampleRows: Record<string, unknown>[]
}

export type Row = Record<string, string | number | null>

export interface AnalysisRequest {
  mode: "smart" | "manual"
  descriptive?: {
    columns: string[]
    measures: ("central" | "spread" | "distribution")[]
  }
  inferential?: {
    correlationPairs?: [string, string][]
    hypothesisTests?: {
      type: "t-test" | "chi-square" | "anova"
      columns: string[]
    }[]
    regression?: {
      dependent: string
      predictors: string[]
    }
  }
  predictive?: {
    dependent: string
    predictors: string[]
    modelType?: ModelType
  }
}

export type ModelType =
  | "linear"
  | "polynomial"
  | "logistic"
  | "multiple"
  | "timeseries"
  | "randomforest"

export interface DescriptiveResult {
  column: string
  mean?: number
  median?: number
  mode?: number | string
  stdDev?: number
  variance?: number
  min?: number
  max?: number
  range?: number
  iqr?: number
  skewness?: number
  kurtosis?: number
  count: number
  nullCount: number
  outlierCount?: number
  frequencyTable?: Record<string, number>
  note?: string
}

export interface TestMetrics {
  rSquared?: number
  rmse?: number
  accuracy?: number
  precision?: number
  recall?: number
  f1?: number
  aucRoc?: number
  sampleSize: number
}

export interface CorrelationResult {
  columnA: string
  columnB: string
  r: number
  method: "pearson" | "spearman"
  interpretation:
    | "strong positive"
    | "moderate positive"
    | "weak"
    | "moderate negative"
    | "strong negative"
}

export interface RegressionResult {
  modelType: ModelType
  dependent: string
  predictors: string[]
  coefficients: number[]
  intercept: number
  rSquared?: number
  adjustedRSquared?: number
  note?: string
  mse?: number
  rmse?: number
  accuracy?: number
  predictions: number[]
  residuals?: number[]
  testPredictions?: number[]
  testMetrics?: TestMetrics
  vif?: { predictor: string; value: number }[]
  featureImportance?: {
    feature: string
    importance: number
  }[]
}

export interface HypothesisResult {
  testType: "t-test" | "chi-square" | "anova"
  statistic: number
  pValue: number
  significant: boolean
  confidenceLevel: number
  columns: string[]
}

export interface InferentialResult {
  correlations?: CorrelationResult[]
  hypothesisTests?: HypothesisResult[]
  regression?: RegressionResult
}

export interface PredictiveResult {
  modelType: ModelType
  regressionResult: RegressionResult
  forecast?: {
    label: string
    predicted: number
    lower: number
    upper: number
  }[]
}

export interface AnalysisResult {
  descriptive?: DescriptiveResult[]
  inferential?: InferentialResult
  predictive?: PredictiveResult
  chartSuggestions: ChartSuggestion[]
}

export type ChartType =
  | "scatter"
  | "line"
  | "bar"
  | "histogram"
  | "heatmap"
  | "boxplot"
  | "pie"
  | "confusion_matrix"
  | "roc_curve"

export interface ChartSuggestion {
  chartType: ChartType
  title: string
  reason: string
  x?: string
  y?: string
  column?: string
  series?: string[]
}

export interface AnalyseRequestBody {
  schema: DatasetSchema
  data: Row[]
  analyses: AnalysisRequest
}

export interface AnalyseResponseBody {
  success: boolean
  result?: AnalysisResult
  error?: string
}

export interface ProfileRequestBody {
  schema: DatasetSchema
}

export interface ProfileResponseBody {
  success: boolean
  output?: ProfilerOutput
  error?: string
}

export interface InterpretRequestBody {
  schema: DatasetSchema
  result: AnalysisResult
}

export interface InterpretResponseBody {
  success: boolean
  summary?: string
  perAnalysis?: {
    type: string
    interpretation: string
  }[]
  error?: string
}

export type AIProvider = "groq" | "mistral" | "gemini" | "huggingface" | "deepseek"

export interface AIResponse {
  content: string
  provider: AIProvider
  fallbackUsed: boolean
}

export interface ProfilerOutput {
  analysisMap: {
    modelType: ModelType
    dependentVariable: string | null
    predictors: string[]
    correlationPairs: [string, string][]
    hypothesisTests: {
      type: "t-test" | "chi-square" | "anova"
      columns: string[]
    }[]
    descriptiveColumns: string[]
  }
  chartSuggestions: ChartSuggestion[]
}

export type MissingValueStrategy =
  | "mean"
  | "median"
  | "mode"
  | "drop_rows"
  | "drop_column"
  | "zero"
  | "forward_fill"
  | "backward_fill"

export type MissingValueStrategyMap = Record<string, MissingValueStrategy>

export interface MissingValueReport {
  totalMissing: number
  byColumn: {
    [columnName: string]: {
      count: number
      percentage: number
      suggestedStrategy: MissingValueStrategy
    }
  }
  requiresAttention: boolean
}

export interface ParsedDataset {
  schema: DatasetSchema
  data: Row[]
  missingValueReport: MissingValueReport
}

export interface PDFSection {
  elementId: string
  title: string
}

export interface PDFGenerationOptions {
  title: string
  fileName: string
  sections: PDFSection[]
  includeTimestamp: boolean
}

export interface PDFGenerationResult {
  success: boolean
  fileName?: string
  error?: string
}