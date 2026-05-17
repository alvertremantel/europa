export type DetailTab = 'attention' | 'activations' | 'logits' | 'network'

export interface PanelShortcutTarget {
  panelId: string
  tab?: DetailTab
}

const PANEL_SHORTCUT_TARGETS = {
  '1': { panelId: 'panel-predictions' },
  '2': { panelId: 'panel-attention', tab: 'attention' },
  '3': { panelId: 'panel-activations', tab: 'activations' },
  '4': { panelId: 'panel-logits', tab: 'logits' },
  '5': { panelId: 'panel-network', tab: 'network' },
} as const satisfies Record<string, PanelShortcutTarget>

export function getPanelShortcutTarget(key: string): PanelShortcutTarget | null {
  return PANEL_SHORTCUT_TARGETS[key as keyof typeof PANEL_SHORTCUT_TARGETS] ?? null
}
