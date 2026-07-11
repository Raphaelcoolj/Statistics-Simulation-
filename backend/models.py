from pydantic import BaseModel, Field
from typing import Optional, Any
from enum import Enum


class ColumnType(str, Enum):
    continuous = "continuous"
    categorical = "categorical"
    ordinal = "ordinal"
    datetime = "datetime"
    binary = "binary"


class Column(BaseModel):
    name: str
    type: ColumnType
    uniqueValues: Optional[list[str | int | float]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    sampleValues: Optional[list[Any]] = None
    nullCount: Optional[int] = None


class DatasetSchema(BaseModel):
    fileName: str
    rowCount: int
    columnCount: int
    columns: list[Column]
    sampleRows: list[dict[str, Any]]


class ModelType(str, Enum):
    linear = "linear"
    polynomial = "polynomial"
    logistic = "logistic"
    multiple = "multiple"
    timeseries = "timeseries"
    randomforest = "randomforest"


class AnalysisRequest(BaseModel):
    mode: str  # "smart" | "manual"
    descriptive: Optional[dict] = None
    inferential: Optional[dict] = None
    predictive: Optional[dict] = None


class DescriptiveResult(BaseModel):
    column: str
    mean: Optional[float] = None
    median: Optional[float] = None
    mode: Optional[float | str] = None
    stdDev: Optional[float] = None
    variance: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    range: Optional[float] = None
    iqr: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    count: int
    nullCount: int
    outlierCount: Optional[int] = None
    frequencyTable: Optional[dict[str, int]] = None
    note: Optional[str] = None


class TestMetrics(BaseModel):
    rSquared: Optional[float] = None
    rmse: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    aucRoc: Optional[float] = None
    sampleSize: int


class CorrelationResult(BaseModel):
    columnA: str
    columnB: str
    r: float
    method: str  # "pearson" | "spearman"
    interpretation: str


class RegressionResult(BaseModel):
    modelType: ModelType
    dependent: str
    predictors: list[str]
    coefficients: list[float]
    intercept: float
    rSquared: Optional[float] = None
    adjustedRSquared: Optional[float] = None
    note: Optional[str] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    accuracy: Optional[float] = None
    predictions: list[float]
    residuals: Optional[list[float]] = None
    testPredictions: Optional[list[float]] = None
    testMetrics: Optional[TestMetrics] = None
    vif: Optional[list[dict]] = None
    featureImportance: Optional[list[dict]] = None


class HypothesisResult(BaseModel):
    testType: str
    statistic: float
    pValue: float
    significant: bool
    confidenceLevel: float
    columns: list[str]


class InferentialResult(BaseModel):
    correlations: Optional[list[CorrelationResult]] = None
    hypothesisTests: Optional[list[HypothesisResult]] = None
    regression: Optional[RegressionResult] = None


class PredictiveResult(BaseModel):
    modelType: ModelType
    regressionResult: RegressionResult
    forecast: Optional[list[dict]] = None
    encodedColumns: Optional[dict[str, dict]] = None  # original_col -> {"encoded": [col_names], "reference": "dropped_category"}


class ChartSuggestion(BaseModel):
    chartType: str
    title: str
    reason: str
    x: Optional[str] = None
    y: Optional[str] = None
    column: Optional[str] = None
    series: Optional[list[str]] = None


class AnalysisResult(BaseModel):
    descriptive: Optional[list[DescriptiveResult]] = None
    inferential: Optional[InferentialResult] = None
    predictive: Optional[PredictiveResult] = None
    chartSuggestions: list[ChartSuggestion]


class MissingValueInfo(BaseModel):
    count: int
    percentage: float
    suggestedStrategy: str


class MissingValueReport(BaseModel):
    totalMissing: int
    byColumn: dict[str, MissingValueInfo]
    requiresAttention: bool


class AnalyseResponse(BaseModel):
    success: bool
    result: Optional[AnalysisResult] = None
    missingValueReport: Optional[MissingValueReport] = None
    schema_: Optional[DatasetSchema] = Field(default=None, alias="schema")
    error: Optional[str] = None
