# AGENTS.md

StatLab: Next.js (14, app router) frontend + Next API routes. Statistical computation runs in a **separate FastAPI Python backend**, not in the route code.

## The most important gotcha: dual backends

- `POST /api/analyse` (`app/api/analyse/route.ts:51`) **proxies all statistics** to a FastAPI service at `PYTHON_BACKEND_URL` (default `http://127.0.0.1:8000`, `app/api/analyse/route.ts:5`). If it's not running you get HTTP 502 "Python backend is not running."
- To run the full stack you need **two** processes:
  - Frontend: `npm run dev` (Next.js)
  - Backend: `npm run dev:backend` (uvicorn `backend.main:app` on port 8000)
- **Do not fix math/statistics bugs in `lib/stats/*.ts`.** Those TS files are now dead in the runtime — only the vitest tests import them. The production path is the Python `backend/stats/*.py` mirror. Fix the Python code.
- The AI routes (`/api/profile`, `/api/interpret`) are still pure Next.js — they call the AI provider chain in `lib/ai/`, not the Python backend.

## Python backend

- Source in `backend/` (`main.py` FastAPI app, `stats/` for computation). No Python tests exist.
- First-time setup: `pip install -r backend/requirements.txt`. Fresh venv recommended.
- Endpoints: `POST /analyse` (multipart: `file`, `analyses` JSON, optional `strategies` JSON) and `GET /health`. CORS allows only `http://localhost:3000`.
- `PYTHON_BACKEND_URL` is not in `.env.local.example`; rely on the default unless the backend runs elsewhere.

## Commands

```bash
npm run dev          # frontend only
npm run dev:backend  # Python backend only (uvicorn, port 8000)
npm test             # vitest (tests the TS lib/stats mirror only)
npm run build        # next build (runs lint + typecheck)
npx tsc --noEmit     # typecheck only
npm run lint         # next lint
```

- Tests are vitest with `@/` aliasing to repo root; no test DB or services required (`npm test` then `npm run build` to verify).

## Environment

- At least one AI key (Groq recommended) in `.env.local` for profiler/interpreter. Stats work with no keys.
- Git history already commits `agents.md` as lowercase alongside `README.md`; keep it.

## Notable past fixes (don't regress)

- Train/test split for held-out metrics must be called once (a double-split bug produced invalid test metrics).
- Multiple regression falls back to ridge (L2) when XᵀX is singular; logistic retries with more steps / stronger L2 if it fails to converge.
- Imputation stats use all rows including rows whose dependent value is later dropped.