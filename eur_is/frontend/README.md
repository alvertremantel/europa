# Europa ALM-IS Frontend

React + Vite frontend for the mechanistic interpretability dashboard.

## Setup

```bash
npm install
```

## Development

Run the backend from the repository root:

```bash
uv run uvicorn eur_is.backend.main:app --reload
```

Then run the frontend from `eur_is/frontend/`:

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
- Full-network Network panel that fetches `/api/analyze` with `include_network=true`
  and shows MLP firing summaries, per-head attention activity, residual-after-attention
  heatmaps, top residual dimensions, and logit-lens projections.

## Notes

- Prompts should use reversed zero-padded arithmetic tokens such as
  `02000000 + 01000000 =`.
- The backend appends `<ans>` automatically before analysis.
- Full-network payload controls are bounded server-side: `mlp_threshold`, `top_k`,
  `top_neurons`, and `selected_token_index`. The frontend requests this payload only
  when the Network panel is opened or refreshed to keep ordinary analysis responses small.
