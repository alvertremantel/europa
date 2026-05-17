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
- 4K-first analysis wall that uses most of a fullscreen television-class monitor,
  with compact/comfortable density controls persisted in local storage.
- Keyboard shortcuts for close-range research use: `/` focuses the prompt, `[` and
  `]` step selected attention layers, and `1`-`5` jump to major analysis panels.
- Checkpoint/device status card sourced from `/api/health`.
- Dense prediction matrix with sticky token columns and top-k confidence bars.
- Attention layer/head matrix with focused CircuitsVis attention view.
- Metric-switchable residual activation heatmap plus CircuitsVis residual browser.
- Generated-answer token timeline and prompt-token logit trajectory summaries.
- Full-network Network panel that fetches `/api/analyze` with `include_network=true`
  and shows MLP firing summaries, per-head attention activity, residual-after-attention
  heatmaps, top residual dimensions, and logit-lens projections.

## 4K visual QA

The primary visual target is fullscreen on a 3840×2160 monitor. When changing layout
or widget styling, review at 3840×2160, 2560×1440, 1920×1080, 960px wide, and a
phone-like narrow viewport. Wide layouts should avoid a page-level horizontal scrollbar;
large tables, CircuitsVis embeds, and network maps may scroll inside their panels.

## Notes

- Prompts should use reversed zero-padded arithmetic tokens such as
  `02000000 + 01000000 =`.
- The backend appends `<ans>` automatically before analysis.
- Full-network payload controls are bounded server-side: `mlp_threshold`, `top_k`,
  `top_neurons`, and `selected_token_index`. The frontend requests this payload only
  when the Network panel is opened or refreshed to keep ordinary analysis responses small.
- CircuitsVis visualizations (`AttentionHeads` and `TextNeuronActivations`) are
  lazy-loaded via `React.lazy()` to keep the ~1.3 MB TensorFlow.js/CircuitsVis
  dependency tree out of the initial application bundle. Each embed region shows a
  compact loading skeleton while the chunk downloads. If a dynamic import fails the
  embed region shows a panel-local error notice — the rest of the panel (matrix,
  heatmap, controls) remains usable.
