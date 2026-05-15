# Europa ALM-IS Frontend

React + Vite frontend for the mechanistic interpretability dashboard.

## Setup

```bash
npm install
```

## Development

Run the backend from the repository root:

```bash
uv run uvicorn web_app.backend.main:app --reload
```

Then run the frontend from `web_app/frontend/`:

```bash
npm run dev
```

Vite proxies `/api/*` requests to `http://localhost:8000`.

## Checks

```bash
npm run lint
npm run build
```

## Dashboard features

- Prompt bar with example arithmetic prompts and Enter-to-analyze behavior.
- Checkpoint/device status card sourced from `/api/health`.
- Compact prediction table with top-5 next-token candidates.
- CircuitsVis attention and residual visualizations embedded in scrollable panels.
- Head summaries, activation norm heatmap, and answer-position logit summaries.

## Notes

- Prompts should use reversed zero-padded arithmetic tokens such as
  `02000000 + 01000000 =`.
- The backend appends `<ans>` automatically before analysis.
