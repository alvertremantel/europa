import assert from 'node:assert/strict'

const modulePath = process.argv[2]
if (!modulePath) {
  throw new Error('usage: node scripts/check-shortcuts.mjs <compiled-keyboard-shortcuts.js>')
}

const { getPanelShortcutTarget } = await import(modulePath)

assert.deepEqual(getPanelShortcutTarget('1'), { panelId: 'panel-predictions' })
assert.deepEqual(getPanelShortcutTarget('2'), { panelId: 'panel-attention', tab: 'attention' })
assert.deepEqual(getPanelShortcutTarget('3'), { panelId: 'panel-activations', tab: 'activations' })
assert.deepEqual(getPanelShortcutTarget('4'), { panelId: 'panel-logits', tab: 'logits' })
assert.deepEqual(getPanelShortcutTarget('5'), { panelId: 'panel-network', tab: 'network' })
assert.equal(getPanelShortcutTarget('6'), null)
assert.equal(getPanelShortcutTarget('/'), null)
