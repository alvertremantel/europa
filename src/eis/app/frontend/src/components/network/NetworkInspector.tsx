import type { NetworkAnalysis } from '../../api'
import type { NetworkSelection } from './types'

interface NetworkInspectorProps {
  network: NetworkAnalysis
  selection: NetworkSelection
}

export function NetworkInspector({ network, selection }: NetworkInspectorProps) {
  return (
    <aside className="network-inspector">
      <h3>Inspector</h3>
      {selection.kind === 'mlp' ? <MlpDetails network={network} selection={selection} /> : null}
      {selection.kind === 'attention' ? <AttentionDetails network={network} selection={selection} /> : null}
      {selection.kind === 'residual' ? <ResidualDetails network={network} selection={selection} /> : null}

      {network.availability.warnings.length > 0 ? (
        <div className="network-warning-list">
          <h4>Cache warnings</h4>
          <ul>
            {network.availability.warnings.slice(0, 6).map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </aside>
  )
}

function MlpDetails({ network, selection }: NetworkInspectorProps & { selection: Extract<NetworkSelection, { kind: 'mlp' }> }) {
  const token = network.mlp.layers[selection.layer]?.tokens[selection.tokenIndex]
  if (!token) {
    return <p>MLP hidden activations are not available for this selection.</p>
  }

  return (
    <div className="network-inspector__body">
      <h4>
        MLP L{selection.layer} · token {selection.tokenIndex} <code>{token.token}</code>
      </h4>
      <dl>
        <Row label="Positive active" value={formatPercent(token.active_fraction_positive)} />
        <Row label="Abs active" value={formatPercent(token.active_fraction_abs)} />
        <Row label="Mean |act|" value={formatNumber(token.mean_abs_activation)} />
        <Row label="Max |act|" value={formatNumber(token.max_abs_activation)} />
        <Row label="Output norm" value={formatNumber(token.output_norm)} />
      </dl>
      <h5>Top neurons</h5>
      <ol>
        {token.top_neurons.map((neuron) => (
          <li key={neuron.neuron_index}>
            <code>n{neuron.neuron_index}</code> {neuron.value.toFixed(4)}
          </li>
        ))}
      </ol>
    </div>
  )
}

function AttentionDetails({
  network,
  selection,
}: NetworkInspectorProps & { selection: Extract<NetworkSelection, { kind: 'attention' }> }) {
  const head = network.attention.layers[selection.layer]?.heads[selection.head]
  if (!head) {
    return <p>Attention activity is not available for this head.</p>
  }

  return (
    <div className="network-inspector__body">
      <h4>
        Attention L{selection.layer} · H{selection.head}
      </h4>
      <dl>
        <Row label="Mean entropy" value={formatNumber(head.mean_entropy)} />
        <Row label="Max weight" value={formatNumber(head.max_weight)} />
        <Row label="Self mass" value={formatNumber(head.self_attention_mass)} />
        <Row label="Previous-token mass" value={formatNumber(head.previous_token_mass)} />
      </dl>
      <p>
        Strongest: token {head.strongest_pair.query_index}{' '}
        <code>{head.strongest_pair.query_token}</code> → token {head.strongest_pair.key_index}{' '}
        <code>{head.strongest_pair.key_token}</code> ({head.strongest_pair.weight.toFixed(3)})
      </p>
    </div>
  )
}

function ResidualDetails({
  network,
  selection,
}: NetworkInspectorProps & { selection: Extract<NetworkSelection, { kind: 'residual' }> }) {
  const token = network.residual.layers[selection.layer]?.tokens[selection.tokenIndex]
  if (!token) {
    return <p>Residual-mid activations are not available for this selection.</p>
  }

  return (
    <div className="network-inspector__body">
      <h4>
        Residual L{selection.layer} · token {selection.tokenIndex} <code>{token.token}</code>
      </h4>
      <dl>
        <Row label="Norm" value={formatNumber(token.norm)} />
        <Row label="Attention delta" value={formatNumber(token.attention_delta_norm)} />
        <Row label="Cosine previous" value={formatNumber(token.cosine_to_previous_mid)} />
        <Row label="Cosine final" value={formatNumber(token.cosine_to_final)} />
      </dl>
      <h5>Logit lens</h5>
      <ol>
        {token.logit_lens_top_k.map((entry) => (
          <li key={entry.token}>
            <code>{entry.token}</code> {(entry.probability * 100).toFixed(2)}% · logit{' '}
            {entry.logit.toFixed(3)}
          </li>
        ))}
      </ol>
    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </>
  )
}

function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return 'not available'
  }
  return value.toFixed(4)
}

function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return 'not available'
  }
  return `${(value * 100).toFixed(2)}%`
}
