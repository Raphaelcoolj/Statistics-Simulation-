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
    coded: Optional[bool] = None
    codeNote: Optional[str] = None
    labels: Optional[dict[str, str]] = None
    codeUncertain: Optional[bool] = None
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
    fullData: Optional[list[dict[str, Any]]] = None
    duplicateRowCount: Optional[int] = None


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
    pValue: Optional[float] = None
    confidenceIntervalLower: Optional[float] = None
    confidenceIntervalUpper: Optional[float] = None


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
    degreesOfFreedom: Optional[int | tuple[int, int]] = None


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
    warnings: Optional[list[str]] = None


class AnalyseResponse(BaseModel):
    success: bool
    result: Optional[AnalysisResult] = None
    missingValueReport: Optional[MissingValueReport] = None
    schema_: Optional[DatasetSchema] = Field(default=None, alias="schema")
    cleaningReport: Optional[CleaningReport] = None
    featureEngineeringReport: Optional[FeatureEngineeringReport] = None
    modelTrainingReport: Optional[ModelTrainingReport] = None
    error: Optional[str] = None


class PreprocessingConfig(BaseModel):
    """Optional preprocessing steps applied before analysis."""
    # Missing data
    missingStrategy: Optional[dict[str, str]] = None  # col -> strategy override
    advancedImputation: Optional[str] = None  # "knn" | "iterative" | None

    # Outlier handling
    outlierColumns: Optional[list[str]] = None
    outlierMethod: Optional[str] = "iqr"  # "iqr" | "zscore"
    outlierAction: Optional[str] = None  # "clip" | "remove" | "log" | "sqrt" | "none"
    outlierFactor: Optional[float] = 1.5

    # Categorical standardization
    standardizeCase: Optional[str] = None  # "lower" | "upper" | "title" | None
    standardizeColumns: Optional[list[str]] = None

    # State standardization
    stateColumns: Optional[list[str]] = None
    stateFormat: Optional[str] = "full"  # "full" | "abbrev"

    # Date parsing
    dateColumns: Optional[list[str]] = None
    parseDates: Optional[bool] = False

    # Typo correction
    typoCorrections: Optional[dict[str, dict[str, str]]] = None  # col -> {wrong: correct}
    typoColumns: Optional[list[str]] = None

    # Deduplication
    removeExactDupes: Optional[bool] = False
    removeFuzzyDupes: Optional[bool] = False
    fuzzyColumns: Optional[list[str]] = None
    fuzzyThreshold: Optional[float] = 0.9


class CleaningReport(BaseModel):
    """Report of all cleaning operations applied."""
    duplicatesRemoved: Optional[dict] = None
    fuzzyDuplicatesRemoved: Optional[dict] = None
    outliersHandled: Optional[dict] = None
    categoricalsStandardized: Optional[dict] = None
    statesStandardized: Optional[dict] = None
    datesParsed: Optional[dict] = None
    typosFixed: Optional[dict] = None
    rowsBefore: Optional[int] = None
    rowsAfter: Optional[int] = None


class FeatureEngineeringConfig(BaseModel):
    """Feature engineering steps applied before predictive modeling."""
    # Encoding
    encodingStrategy: Optional[str] = None  # "auto" | "onehot" | "target" | "ordinal"
    encodingColumns: Optional[list[str]] = None
    ordinalMaps: Optional[dict[str, list[str]]] = None  # col -> ordered values

    # Scaling
    scalingMethod: Optional[str] = None  # "standard" | "minmax"
    scalingColumns: Optional[list[str]] = None
    scalingExclude: Optional[list[str]] = None  # cols to skip (e.g. dependent)

    # Feature creation
    datetimeColumns: Optional[list[str]] = None
    datetimeFeatures: Optional[list[str]] = None
    ratioPairs: Optional[list[tuple[str, str]]] = None
    interactionPairs: Optional[list[tuple[str, str]]] = None
    aggregationColumns: Optional[list[str]] = None
    aggregationGroupBy: Optional[str] = None

    # Feature selection
    removeCorrelated: Optional[bool] = False
    correlationThreshold: Optional[float] = 0.95
    targetCorrelationFilter: Optional[bool] = False
    targetCorrelationThreshold: Optional[float] = 0.05
    applyVifFilter: Optional[bool] = False
    vifThreshold: Optional[float] = 10.0
    applyLassoSelection: Optional[bool] = False
    lassoAlpha: Optional[float] = 0.01
    applyPca: Optional[bool] = False
    pcaVarianceThreshold: Optional[float] = 0.95
    applyFeatureImportance: Optional[bool] = False
    featureImportanceTopK: Optional[int] = None


class FeatureEngineeringReport(BaseModel):
    """Report of all feature engineering operations applied."""
    encoding: Optional[dict] = None
    scaling: Optional[dict] = None
    datetimeFeatures: Optional[dict] = None
    ratioFeatures: Optional[list[str]] = None
    interactionFeatures: Optional[list[str]] = None
    aggregationFeatures: Optional[list[str]] = None
    correlatedFilter: Optional[dict] = None
    targetCorrelationFilter: Optional[dict] = None
    vifFilter: Optional[dict] = None
    lassoSelection: Optional[dict] = None
    pcaResult: Optional[dict] = None
    featureImportanceSelection: Optional[dict] = None
    columnsBefore: Optional[int] = None
    columnsAfter: Optional[int] = None


class ModelTrainingConfig(BaseModel):
    """Configuration for the ML model training/tuning/evaluation run."""
    enabled: Optional[bool] = False
    problemType: Optional[str] = None            # "classification" | "regression" (auto if omitted)
    models: Optional[list[str]] = None           # e.g. ["random_forest", "xgboost", "lightgbm", "neural_network"]
    tuningMethod: Optional[str] = "random"       # "grid" | "random" | "bayesian" | "none"
    tuningIterations: Optional[int] = 15
    cvFolds: Optional[int] = 5
    testSize: Optional[float] = 0.2
    valSize: Optional[float] = 0.15
    randomSeed: Optional[int] = 42


class ModelResult(BaseModel):
    """Per-model training result."""
    name: Optional[str] = None
    tuning: Optional[dict] = None
    trainMetrics: Optional[dict] = None
    valMetrics: Optional[dict] = None
    testMetrics: Optional[dict] = None
    crossValidation: Optional[dict] = None
    testMape: Optional[float] = None
    error: Optional[str] = None


class ModelTrainingReport(BaseModel):
    """Report of the full ML training/tuning/evaluation experiment."""
    problemType: Optional[str] = None
    split: Optional[dict] = None
    baselines: Optional[dict] = None
    models: Optional[dict] = None
    bestModel: Optional[dict] = None
    explainability: Optional[dict] = None
    businessTranslation: Optional[dict] = None
    featureInsights: Optional[dict] = None
    recommendations: Optional[list] = None
    charts: Optional[dict] = None
    error: Optional[str] = None
