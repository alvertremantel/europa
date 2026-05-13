# Handoff: MLP Neurons Tab Blank View in web_app

**Date:** 2026-05-13  
**Status:** Regression introduced, needs revert + fresh approach

## The Problem

The MLP Neurons tab in the MathAdder web_app (`web_app/frontend/src/App.tsx`) renders a blank/white view when selected. The other three tabs (Attention, Residual Stream, Logit Lens) work fine.

## What Was Tried

### Attempt 1 (reverted): Add `import 'plotly.js-dist-min'`
Added a side-effect import of `plotly.js-dist-min` before `import Plot from 'react-plotly.js'`. Rationale: `react-plotly.js` might need `window.Plotly` to be set globally. **Result:** MLP Neurons tab still white.

### Attempt 2 (current, broken): Use factory API directly
Changed from:
```ts
import Plot from 'react-plotly.js';
```
To:
```ts
import Plotly from 'plotly.js-dist-min';
import createPlotlyComponent from 'react-plotly.js/factory';
const Plot = createPlotlyComponent(Plotly);
```
Rationale: `react-plotly.js` internally does `require("plotly.js/dist/plotly")` which, after Vite's CJS-to-ESM interop, may double-unwrap the default export resulting in `undefined`. Using the factory directly would bypass this.  
**Result:** ENTIRE app goes white — this broke everything, not just MLP Neurons.

## Key Findings

1. **Backend data is valid** — verified via `uv run` test:
   - `mlp.hook_post` exists in cache for all 6 layers
   - Shape: `(25, 512)` = `(tokens, mlp_hidden)`
   - Values: min ~-0.17, max ~5.24, no NaNs/infs
   - Config: n_layers=6, d_mlp=512, d_model=128, n_heads=2

2. **Dependency landscape:**
   - `react-plotly.js` v2.6.0 — peer depends on `plotly.js > 1.34.0`
   - `plotly.js` v3.5.1 (full) — installed as transitive/peer dep, used internally by react-plotly.js
   - `plotly.js-dist-min` v3.5.1 — separate smaller bundle, NOT used by react-plotly.js by default

3. **react-plotly.js internals:**
   - Entry file (`react-plotly.js`) does: `var _plotly = require("plotly.js/dist/plotly"); var PlotComponent = _factory(_plotly["default"]);`
   - Factory receives a pre-resolved Plotly object
   - `plotly.js-dist-min` Node ESM import works fine in isolation

4. **The factory import path** `react-plotly.js/factory` **exists** and has TypeScript types (`@types/react-plotly.js/factory.d.ts`)

## Current File State

`web_app/frontend/src/App.tsx` — lines 1-6 currently:
```ts
import React, { useState } from 'react';
import axios from 'axios';
import { AttentionHeads, TextNeuronActivations } from 'circuitsvis';
import Plotly from 'plotly.js-dist-min';
import createPlotlyComponent from 'react-plotly.js/factory';
const Plot = createPlotlyComponent(Plotly);
```

## Immediate Next Steps

1. **Revert to original working state** — restore `import Plot from 'react-plotly.js'` to get the app working again (minus MLP Neurons tab).

2. **Fresh investigation for MLP Neurons fix:**
   - Open browser DevTools console while on the MLP Neurons tab to catch any JS errors
   - Add `onError={(err) => console.error('Plotly error:', err)}` to the `<Plot>` component to surface silent errors
   - Add `onInitialized={(fig, el) => console.log('Plotly initialized', fig, el)}` to confirm if Plotly even runs
   - Check if the `<Plot>` component's container div has non-zero dimensions at render time
   - Consider wrapping Plot in an error boundary to prevent crashes from affecting other tabs

3. **Alternative approaches to investigate:**
   - Try a simpler heatmap first (hardcoded sample data) to isolate data vs. library issues
   - Check if `plotly.js/dist/plotly` (the full one react-plotly.js uses) is properly bundled by Vite — look at the built JS for `Plotly.react`
   - Consider using the full `plotly.js` package instead of `plotly.js-dist-min` if needed
   - Try `react-plotly.js` v2.x recommended pattern: `import Plotly from 'plotly.js-dist-min'` then `<Plot data={...} layout={...} />` (might need `useResizeHandler` prop)

## Revert Command
```bash
cd /mnt/ssd/work/europa && git checkout web_app/frontend/src/App.tsx
```
