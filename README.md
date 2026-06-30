# StatLab

Intelligent statistical analysis platform. Upload a CSV, get instant descriptive/inferential/predictive analysis, AI-powered plain-English interpretation, and a downloadable PDF report.

Built by the **Department of Statistics, University of Benin** for the **Physical Science Innovation Competition 2026**.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Backend — API Routes](#backend--api-routes)
  - [POST /api/profile](#1-post-apiprofile--ai-profiler)
  - [POST /api/analyse](#2-post-apianalyse--core-computation)
  - [POST /api/interpret](#3-post-apiinterpret--ai-interpretation)
- [Backend — Statistical Engine](#backend--statistical-engine)
  - [Parser (`lib/stats/parser.ts`)](#parser)
  - [Descriptive Statistics (`lib/stats/descriptive.ts`)](#descriptive-statistics)
  - [Inferential Statistics (`lib/stats/inferential.ts`)](#inferential-statistics)
  - [Preprocessing (`lib/stats/preprocessing.ts`)](#preprocessing)
  - [Predictive Modeling (`lib/stats/predictive.ts`)](#predictive-modeling)
- [Backend — AI Layer](#backend--ai-layer)
- [Backend — PDF Generation](#backend--pdf-generation)
- [Frontend — Pages & Components](#frontend--pages--components)
- [Frontend — Data Flow](#frontend--data-flow)
- [Frontend — Chart Rendering](#frontend--chart-rendering)
- [Types Reference](#types-reference)
- [Environment Variables](#environment-variables)

---

## Overview

StatLab provides a complete statistical analysis pipeline in a single-page web application:

```
Upload CSV → Parse & detect types → Profile (AI or manual) → Compute statistics
→ Interpret with AI → Render charts + tables + text → Export PDF
```

**Key features:**

- **No database, no user accounts.** All computation is in-memory; history is stored in localStorage. Zero infrastructure costs.
- **Stateless backend.** The Python-free JS stack runs entirely in Next.js API routes.
- **Multi-model predictive engine.** Auto-selects from linear, polynomial, multiple, logistic, timeseries, and random forest models. Handles multicollinearity with ridge regression fallback, missing values with mean/median/mode imputation, and categoricals with one-hot encoding.
- **AI analysis.** A fallback chain of 5 AI providers (Groq → Mistral → Gemini → DeepSeek → HuggingFace) generates human-readable interpretations. Never fails — returns a generic summary if all providers are down.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Browser (Next.js)                 │
│                                                      │
│  ┌──────────┐   ┌──────────────┐   ┌─────────────┐  │
│  │ Upload   │──▶│ Analyse Page │──▶│ PDF Export  │  │
│  │ Page (/) │   │ (/analyse)   │   │ (client)    │  │
│  └──────────┘   └──────┬───────┘   └─────────────┘  │
│                         │                            │
│                    POST │ multipart                   │
│                    JSON │ / JSON                     │
└─────────────────────────┼────────────────────────────┘
                          │
┌─────────────────────────┼────────────────────────────┐
│              Next.js API Routes (Server)              │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ /api/profile │  │ /api/analyse │  │ /api/      │ │
│  │ (AI profiler)│  │ (computation)│  │  interpret │ │
│  └──────┬───────┘  └──────┬───────┘  │ (AI text)  │ │
│         │                 │           └────────────┘ │
│         ▼                 ▼                           │
│  ┌────────────────────────────────────────────┐      │
│  │         lib/ (shared logic)                │      │
│  │  ┌────────┐ ┌──────────┐ ┌─────────────┐  │      │
│  │  │parser  │ │descriptive│ │ inferential │  │      │
│  │  │.ts     │ │.ts       │ │ .ts         │  │      │
│  │  ├────────┤ ├──────────┤ ├─────────────┤  │      │
│  │  │pre-    │ │predictive│ │ provider-   │  │      │
│  │  │process │ │.ts       │ │ Chain.ts    │  │      │
│  │  └────────┘ └──────────┘ └─────────────┘  │      │
│  └────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────┘
```

**Design principles:**

- **Computation never depends on AI.** The analysis engine (`lib/stats/`) runs pure math — no AI calls, no external dependencies. AI is used only for interpretation and profiling.
- **Graceful degradation.** Every layer has a fallback: missing values get imputed, singular matrices trigger ridge regression, AI failures return generic results, localStorage failures don't crash the app.

---

## Quick Start

```bash
# Install
npm install

# Configure environment — at least one AI key required
cp .env.local.example .env.local
# Edit .env.local with your API keys (Groq recommended — free tier, fastest)

# Run
npm run dev       # http://localhost:3000

# Build for production
npm run build
npm start

# Run tests
npm test          # vitest
```

---

## Backend — API Routes

### 1. `POST /api/profile` — AI Profiler

Smart Analyse entry point. Sends the dataset schema to the AI profiler, which recommends what analyses and charts to run.

**Request:** `application/json`
```json
{ "schema": { "fileName": "titanic.csv", "rowCount": 891, "columns": [...], "sampleRows": [...] } }
```

**Response:**
```json
{
  "success": true,
  "output": {
    "analysisMap": {
      "modelType": "logistic",
      "dependentVariable": "survived",
      "predictors": ["pclass", "sex", "age", "fare"],
      "correlationPairs": [["age", "fare"]],
      "hypothesisTests": [],
      "descriptiveColumns": ["age", "fare", "pclass", "sex", "survived"]
    },
    "chartSuggestions": [
      { "chartType": "histogram", "title": "Age Distribution", "column": "age", "reason": "..." },
      { "chartType": "scatter", "title": "Age vs Fare", "x": "age", "y": "fare", "reason": "..." }
    ],
    "relationshipSuggestions": [
      { "dependent": "survived", "predictors": ["pclass", "sex", "age", "fare"], "modelType": "logistic", "reason": "..." }
    ]
  }
}
```

**Fallback:** If AI fails or returns unparseable JSON, returns a rule-based safe default that runs descriptive stats and histograms on all continuous columns.

**Rate limited:** 20 requests per minute per IP.

---

### 2. `POST /api/analyse` — Core Computation

Accepts the CSV file and analysis request. Runs the full statistical pipeline.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | CSV file (max 50 MB) |
| `analyses` | JSON string | Yes | `AnalysisRequest` — what to compute |
| `strategies` | JSON string | No | `MissingValueStrategyMap` — per-column imputation strategy |

**AnalysisRequest structure:**
```json
{
  "mode": "smart",
  "descriptive": { "columns": ["age", "fare", "sex"], "measures": ["central", "spread", "distribution"] },
  "inferential": {
    "correlationPairs": [["age", "fare"]],
    "hypothesisTests": [{ "type": "t-test", "columns": ["age", "survived"] }]
  },
  "predictive": { "dependent": "survived", "predictors": ["pclass", "sex", "age"], "modelType": "logistic" }
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "descriptive": [ { "column": "age", "mean": 29.7, "median": 28, "stdDev": 14.5, ... } ],
    "inferential": {
      "correlations": [ { "columnA": "age", "columnB": "fare", "r": 0.18, "method": "pearson", "interpretation": "weak" } ],
      "hypothesisTests": [ { "testType": "t-test", "statistic": 2.34, "pValue": 0.02, "significant": true } ]
    },
    "predictive": {
      "modelType": "logistic",
      "regressionResult": {
        "coefficients": [-1.2, 2.6, -0.04],
        "intercept": 3.8,
        "accuracy": 0.81,
        "note": "R² not applicable. Accuracy: 81.0%",
        ...
      }
    }
  },
  "schema": { "fileName": "titanic.csv", "rowCount": 891, ... },
  "missingValueReport": { "totalMissing": 177, "byColumn": { "age": { "count": 177, "percentage": 19.9, "suggestedStrategy": "mean" } }, "requiresAttention": true }
}
```

---

### 3. `POST /api/interpret` — AI Interpretation

Takes computed results and generates plain-English explanations.

**Request:** `application/json`
```json
{ "schema": {...}, "result": { "descriptive": [...], "inferential": {...}, "predictive": {...} } }
```

**Response:**
```json
{
  "success": true,
  "summary": "The data shows that passengers who paid more for their tickets were more likely to survive...",
  "perAnalysis": [
    { "type": "predictive", "subject": "survived ~ pclass + sex + age", "interpretation": "The logistic model achieved 81% accuracy..." }
  ],
  "provider": "groq",
  "fallbackUsed": false
}
```

**Critical:** This route **never returns `success: false`**. If all AI providers are down, it returns a generic summary. Always read `summary` as optional.

---

## Backend — Statistical Engine

### Parser

**File:** `lib/stats/parser.ts`

Parses CSV files using `csv-parse/sync`. Runs in two phases:

1. **Type detection** — For each column, samples values to determine if it's `continuous`, `categorical`, `ordinal`, `datetime`, or `binary`. Binary detection checks for exactly 2 unique values (0/1, true/false, yes/no).
2. **Missing value analysis** — Scans every column, computes null counts per column, and suggests an imputation strategy per column type (mean for continuous, median for ordinal, mode for categorical, drop for excessive missing).

**Key exported functions:**

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `parseCSV(buffer, fileName)` | Raw CSV buffer, file name | `{ schema, data, missingValueReport }` | Full parse pipeline |
| `detectMissingValues(data, columns)` | Parsed rows, column defs | `MissingValueReport` | Per-column null counts and strategy suggestions |
| `applyMissingValueStrategy(rows, columns, strategies)` | Rows, columns, strategy map | Cleaned rows | Applies chosen imputation strategy |

### Descriptive Statistics

**File:** `lib/stats/descriptive.ts`

Computes univariate statistics using `simple-statistics`. Type-aware — different measures for different column types.

| Column Type | Computed Fields |
|-------------|-----------------|
| `continuous`, `ordinal` | mean, median, mode, stdDev, variance, min, max, range, IQR, skewness, kurtosis, outlierCount |
| `categorical`, `binary` | mode, frequencyTable (value → count map) |
| `datetime` | min, max, range (as timestamps) |

### Inferential Statistics

**File:** `lib/stats/inferential.ts`

| Feature | Function | Details |
|---------|----------|---------|
| **Correlation** | `computeCorrelations(pairs, data, columns)` | Auto-selects Pearson (continuous × continuous), Spearman (ordinal × continuous), or point-biserial (binary × continuous). Labels: strong positive (r>0.7), moderate positive (0.4<r<0.7), weak (|r|<0.4), moderate negative, strong negative. |
| **T-Test** | `runTTest(colA, colB, data)` | Two-sample Welch's t-test. Reports statistic, p-value, significance at 95% confidence. |
| **Chi-Square** | `runChiSquare(colA, colB, data)` | Tests independence between two categorical columns. |
| **ANOVA** | `runANOVA(dependent, groupCol, data)` | One-way ANOVA testing if group means differ. |

### Preprocessing

**File:** `lib/stats/preprocessing.ts`

Runs before any predictive model to prepare the data.

| Pipeline Step | Function | What it does |
|---------------|----------|-------------|
| 1. Imputation | `applyMissingValueImputation(data, columns)` | Fills missing predictor values using all available rows: mean for continuous, median for ordinal, mode for categorical/binary |
| 2. Row filtering | _(inline in preprocessForModel)_ | Drops rows where the dependent variable is still missing after imputation |
| 3. One-hot encoding | `oneHotEncode(data, columns)` | Converts categorical/binary columns to 0/1 dummy columns. Drops the first category to avoid multicollinearity (dummy variable trap). |
| 4. Standardization | `applyStandardization(data, params)` | Z-score normalizes continuous predictors: `(x - mean) / std` |
| 5. Train/test split | `trainTestSplit(data)` | 80/20 random split. Uses a single call (fixed from a previous bug where split was called twice, producing invalid test metrics). |

**Additional utilities:**

| Function | Description |
|----------|-------------|
| `computeVIF(data, predictors, columns)` | Variance Inflation Factor for multicollinearity detection (VIF > 5 flags high collinearity) |
| `detectOutliersIQR(data, column)` | Flags values outside Q1 - 1.5×IQR / Q3 + 1.5×IQR |

### Predictive Modeling

**File:** `lib/stats/predictive.ts`

The most complex module. Orchestrates model selection, training, and evaluation.

#### Model Selection (`selectModel`)

Auto-selects the best model type based on the data:

| Condition | Selected Model |
|-----------|---------------|
| Dependent is binary | `logistic` |
| Multiple predictors (>1) | `multiple` |
| Has datetime predictor | `timeseries` |
| Low linear R² (< 0.6) with 1 predictor | `polynomial` (tests if polynomial improves R² by >0.1) |
| 1 continuous predictor | `linear` |
| Large data (>500 rows) or many predictors (>5) | `randomforest` |

#### Model Implementations

| Model | Function | JS Library | Key Improvements |
|-------|----------|------------|-----------------|
| Linear | `runLinearRegression` | simple-statistics | OLS via `ss.linearRegression` |
| Polynomial | `runPolynomialRegression` | ml-regression `PolynomialRegression` | Degree-2 polynomial fit |
| Multiple | `runMultipleRegression` | Custom normal equations | **Ridge regression fallback** — if XᵀX is singular, adds L2 penalty (λ=0.001, or λ=1 for many predictors) to guarantee invertibility instead of falling back to 1-predictor linear |
| Logistic | `runLogisticRegression` | ml-logistic-regression | 5000 training steps, L2 regularization (λ=0.01), retries with 10000 steps + λ=0.1 if first attempt fails |
| Timeseries | `runTimeSeriesRegression` | simple-statistics + custom | Linear regression on time-ordered data. Generates 5-period forecast with 95% confidence intervals. |
| Random Forest | `runRandomForest` | ml-random-forest | Ensemble of 200 trees. Returns feature importance scores. |

#### Evaluation Metrics

All models compute:
- **Predictions** — fitted values for training data
- **Residuals** — actual − predicted
- **RMSE** — root mean squared error
- **R²** — coefficient of determination (not applicable for logistic)

When data >= 30 rows, an 80/20 train/test split is used to compute held-out metrics:
- **Test R²** — generalization performance (regression)
- **Test RMSE** — generalization error
- **Accuracy, Precision, Recall, F1, AUC-ROC** — classification metrics (logistic)

#### Missing Value Handling

- **Predictor imputation** uses all rows (not just rows with valid dependent values) to compute imputation statistics — producing more accurate mean/mode estimates.
- **Dependent variable** rows with missing target values are dropped only after imputation (they still contributed to imputation statistics).

---

## Backend — AI Layer

**File:** `lib/ai/providerChain.ts`

Implements a fallback chain that tries AI providers in order until one succeeds:

```
Groq (llama-3.1-8b-instant) → Mistral (open-mistral-7b) → Gemini (gemini-1.5-flash)
→ HuggingFace (mistral-7b-instruct) → DeepSeek (deepseek-chat)
```

**Key function:** `callAI(systemPrompt, userPrompt) → { content, provider, fallbackUsed }`

Each response is validated against expected JSON schema. If parsing fails, the chain falls through to the next provider. If all providers fail, the error propagates to the caller for graceful fallback handling.

**Two AI roles:**

1. **Profiler** (`lib/ai/profilerPrompt.ts`) — Given a dataset schema, recommends analysis configuration and chart suggestions. Called by `POST /api/profile`.
2. **Interpreter** (`lib/ai/interpreterPrompt.ts`) — Given computed results, produces plain-English explanations. Called by `POST /api/interpret`.

---

## Backend — PDF Generation

**Files:** `lib/pdf/generator.ts`, `lib/pdf/usePDFExport.ts`

Fully client-side. Uses `html2canvas` to screenshot DOM elements and `jsPDF` to assemble them into a PDF.

```ts
const { exportPDF, isGenerating } = usePDFExport()

await exportPDF({
  title: 'Titanic Dataset Analysis',
  fileName: 'statlab-report.pdf',
  includeTimestamp: true,
  sections: [
    { elementId: 'descriptive-stats', title: 'Descriptive Statistics' },
    { elementId: 'charts-panel', title: 'Charts & Visualisations' },
    { elementId: 'ai-interpretation', title: 'AI Interpretation' },
  ],
})
```

Each section must have an HTML `id` attribute matching the `elementId`. A cover page is rendered programmatically with jsPDF; content pages are screenshots of the live DOM.

---

## Frontend — Pages & Components

### Pages

| Route | File | Description |
|-------|------|-------------|
| `/` | `app/page.tsx` | CSV upload page. Handles file selection, mode toggle (Smart/Manual), and initiates the analysis pipeline. |
| `/analyse` | `app/analyse/page.tsx` | Full results page (~1200 lines). Renders all charts, statistics tables, model cards, AI interpretations, and PDF export. |

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| `StatLabProvider` | `components/StatLabProvider.tsx` | React context provider wrapping the entire app. Exposes the `useStatLab()` hook for state management across all pages. |
| `ErrorBoundary` | `components/ErrorBoundary.tsx` | Class-based error boundary. Catches rendering errors in chart components and shows a fallback UI. Prevents single chart failures from crashing the entire page. |
| `ChartCard` | `app/analyse/page.tsx` (inline) | Renders individual chart suggestions using Recharts. Handles all chart types with smart fallback logic — shows "Chart not available — insufficient data" when data doesn't support the requested chart type. |
| `RelationshipPanel` | `app/analyse/page.tsx` (inline) | Predictive modeling dashboard. Shows model overview cards, example predictions, coefficient charts, VIF indicators, feature importance bars, and predicted-vs-actual scatter plots. |

### Custom Hooks

| Hook | File | Description |
|------|------|-------------|
| `useStatLab()` | `lib/useStatLab.ts` | **Primary state manager.** Manages: API pipeline (idle→parsing→profiling→analysing→interpreting→done), file/schema state, session history (localStorage, max 20 sessions), mode switching, and exposes `submit()`, `loadSession()`, `reset()`, `runPredictiveModel()`. |
| `usePDFExport()` | `lib/pdf/usePDFExport.ts` | Wraps the PDF generator. Exposes `exportPDF()`, `isGenerating` loading state, and `error` state. |

---

## Frontend — Data Flow

```
┌──────────────────────────────────────────────────────────┐
│                    Smart Analyse Flow                     │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  1. User selects CSV → client-side parse with PapaParse   │
│     → build DatasetSchema (column names, types, sample    │
│     rows)                                                  │
│                                                           │
│  2. POST schema → /api/profile                             │
│     → returns analysisMap + chartSuggestions               │
│                                                           │
│  3. Map analysisMap → AnalysisRequest                      │
│     → POST file + AnalysisRequest → /api/analyse           │
│     → returns AnalysisResult (numbers, no AI text)         │
│                                                           │
│  4. POST schema + AnalysisResult → /api/interpret          │
│     → returns summary + perAnalysis (plain English)        │
│                                                           │
│  5. Render everything:                                     │
│     - Charts from chartSuggestions + descriptive data      │
│     - Tables from result.descriptive / .inferential        │
│     - Model cards from result.predictive                   │
│     - AI text from interpret response                      │
│                                                           │
│  6. User clicks "Export PDF" → client-side PDF generation  │
│     (html2canvas screenshots + jsPDF assembly)             │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

**Manual mode** skips step 2 (profile). The user builds an `AnalysisRequest` directly from form inputs, then goes straight to step 3.

---

## Frontend — Chart Rendering

**Library:** Recharts

All charts are rendered inside a `ChartCard` component that handles data processing and chart type selection:

| Chart Type | Data Source | Rendering |
|------------|-------------|-----------|
| **Bar** | `frequencyTable` from descriptive stats | `<BarChart>` with count values |
| **Pie** | `frequencyTable` (max 6 slices) | `<PieChart>` with colored cells |
| **Line** | `frequencyTable` | `<LineChart>` with continuous line |
| **Scatter** | Full dataset (all rows, sampled to max 2000 points) | `<ScatterChart>` with x/y columns from suggestion |
| **Histogram** | Descriptive stats (min, max, mean, stdDev) | `<BarChart>` with 10 estimated bins |
| **Heatmap** | Correlation matrix | Custom grid layout with color-coded cells |
| **Boxplot** | Descriptive stats (min, q1, median, q3, max) | Custom CSS-rendered box plot |

**Fallback behavior:** If a chart type is requested but the required data isn't available, ChartCard displays "Chart not available — insufficient data" instead of showing misleading data (e.g., no more "Row 1, Row 2..." labels).

---

## Types Reference

All types are in `lib/types.ts`. Key types:

| Type | Description |
|------|-------------|
| `Column` | Name, type (continuous/categorical/ordinal/datetime/binary), stats, unique values |
| `DatasetSchema` | File metadata + column definitions + sample rows (all rows) |
| `ModelType` | linear \| polynomial \| logistic \| multiple \| timeseries \| randomforest |
| `DescriptiveResult` | Per-column: mean, median, mode, stdDev, min, max, skewness, frequency table, etc. |
| `CorrelationResult` | Pairwise: columns, r value, method (pearson/spearman), interpretation label |
| `RegressionResult` | Coefficients, intercept, R², RMSE, predictions, residuals, VIF, feature importance |
| `TestMetrics` | rSquared, rmse, accuracy, precision, recall, f1, aucRoc |
| `PredictiveResult` | modelType + regressionResult + optional forecast |
| `ChartSuggestion` | chartType + title + reason + optional x/y/column/series |
| `RelationshipSuggestion` | dependent + predictors + modelType + reason |
| `MissingValueStrategy` | mean \| median \| mode \| drop_rows \| drop_column \| zero \| forward_fill \| backward_fill |

---

## Environment Variables

| Variable | Required | Provider | Notes |
|----------|----------|----------|-------|
| `GROQ_API_KEY` | Recommended | [groq.com](https://groq.com) | Fastest, generous free tier — preferred first provider |
| `MISTRAL_API_KEY` | Optional | [mistral.ai](https://mistral.ai) | Fallback if Groq fails |
| `GEMINI_API_KEY` | Optional | [ai.google.dev](https://ai.google.dev) | Fallback if Mistral fails |
| `DEEPSEEK_API_KEY` | Optional | [deepseek.com](https://deepseek.com) | Fallback |
| `HUGGINGFACE_API_KEY` | Optional | [huggingface.co](https://huggingface.co) | Last fallback before generic summary |

At least one key must be set for AI features (profiling and interpretation) to work. The analysis engine (`lib/stats/`) runs without any API keys — it's pure math.

---

## File Tree

```
├── app/
│   ├── analyse/page.tsx          # Analysis results page (~1200 loc)
│   ├── api/
│   │   ├── analyse/route.ts      # POST — core computation
│   │   ├── interpret/route.ts    # POST — AI interpretation
│   │   └── profile/route.ts      # POST — AI profiling
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx                  # Upload page
├── components/
│   ├── ErrorBoundary.tsx
│   └── StatLabProvider.tsx
├── lib/
│   ├── ai/
│   │   ├── providerChain.ts      # AI fallback (Groq→Mistral→Gemini→...)
│   │   ├── profilerPrompt.ts     # Profiler prompts + response parsing
│   │   └── interpreterPrompt.ts  # Interpreter prompts + response parsing
│   ├── pdf/
│   │   ├── generator.ts          # jsPDF + html2canvas assembly
│   │   └── usePDFExport.ts       # React hook for PDF export
│   ├── stats/
│   │   ├── descriptive.ts        # Mean, median, stdDev, frequency tables
│   │   ├── inferential.ts        # Correlations, t-tests, chi-square, ANOVA
│   │   ├── parser.ts             # CSV parsing + type detection + missing values
│   │   ├── predictive.ts         # All 6 model types + ridge regression fallback
│   │   └── preprocessing.ts      # Imputation, encoding, standardization
│   ├── types.ts                  # Central type definitions
│   ├── config.ts                 # Environment variable management
│   ├── useDebounce.ts            # Generic debounce hook
│   ├── useStatLab.ts             # Primary state management hook
│   └── utils/
│       ├── errors.ts             # Error classes + formatting
│       ├── validation.ts         # Request validation (file, schema, analysis)
│       ├── rateLimit.ts          # In-memory rate limiter
│       └── schemas.ts            # Zod validation schemas
├── tests/
│   ├── descriptive.test.ts
│   ├── predictive.test.ts
│   └── preprocessing.test.ts
└── docs/
    ├── API.md                    # Full API reference
    ├── FRONTEND.md               # Frontend integration guide
    ├── ROUTES.md                 # Route map
    └── TYPES.md                  # Type documentation
```
