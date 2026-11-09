import { useId, useState, type JSX } from 'react';
import { computePolicyScore, computeSelectionScore, explorationCoefficient, selectionNoise, type Legality } from './policyScore';
import PolicyScoreSurface from './PolicyScoreSurface';
import './PuctDemo.css';
import './PolicyScoreDemo.css';

const initial = { probability: 0.6, nodeVisits: 32, actionVisits: 8, normalizedValue: 0.5, legality: 'legal' as Legality };

export default function PolicyScoreDemo(): JSX.Element {
  const id = useId();
  const [state, setState] = useState(initial);
  const [maxVisits, setMaxVisits] = useState(64);
  const { probability, nodeVisits, actionVisits, normalizedValue, legality } = state;
  const coefficient = explorationCoefficient(nodeVisits);
  const policyScore = computePolicyScore(probability, nodeVisits, actionVisits);
  const valueScore = actionVisits > 0 ? normalizedValue : 0;
  const score = computeSelectionScore(probability, nodeVisits, actionVisits, normalizedValue, legality);
  const scoreLabel = score === -Infinity ? '−∞' : score.toFixed(3);

  return (
    <figure className="puct-demo policy-score-demo" aria-labelledby={`${id}-title`}>
      <div className="puct-heading">
        <strong id={`${id}-title`}>The final score before argmax</strong>
        <button type="button" onClick={() => { setState(initial); setMaxVisits(64); }}>Reset</button>
      </div>
      <div className="policy-score-controls">
        <label htmlFor={`${id}-probability`}>Action probability <code>P(a)</code><output>{probability.toFixed(2)}</output>
          <input id={`${id}-probability`} aria-label="Action probability" type="range" min="0" max="1" step="0.01" value={probability}
            onChange={(event) => setState({ ...state, probability: event.currentTarget.valueAsNumber })} />
          <small>The remaining probability belongs to other actions.</small>
        </label>
        <label htmlFor={`${id}-node-visits`}>Node visits <code>N</code><output>{nodeVisits}</output>
          <input id={`${id}-node-visits`} aria-label="Node visits" type="range" min="1" max={maxVisits} step="1" value={nodeVisits}
            onChange={(event) => {
              const visits = event.currentTarget.valueAsNumber;
              setState({ ...state, nodeVisits: visits, actionVisits: Math.min(actionVisits, visits - 1) });
            }} />
          <small>Includes the node’s initial prediction as visit 1.</small>
        </label>
        <label htmlFor={`${id}-action-visits`}>Visits through this action <code>n</code><output>{actionVisits}</output>
          <input id={`${id}-action-visits`} aria-label="Action visits" type="range" min="0" max={Math.max(1, nodeVisits - 1)} step="1" value={actionVisits} disabled={nodeVisits === 1}
            onChange={(event) => setState({ ...state, actionVisits: event.currentTarget.valueAsNumber })} />
          <small>{nodeVisits - 1 - actionVisits} visits through other actions.</small>
        </label>
        <label htmlFor={`${id}-value`}>Normalized value <code>Q̄</code><output>{normalizedValue.toFixed(2)}</output>
          <input id={`${id}-value`} aria-label="Normalized value" type="range" min="0" max="1" step="0.01" value={normalizedValue}
            onChange={(event) => setState({ ...state, normalizedValue: event.currentTarget.valueAsNumber })} />
          <small>Value after normalization against the parent and visited siblings; unvisited actions use zero.</small>
        </label>
        <label htmlFor={`${id}-legality`}>Legal-action mask
          <select id={`${id}-legality`} aria-label="Legal-action mask" value={legality}
            onChange={(event) => setState({ ...state, legality: event.currentTarget.value as Legality })}>
            <option value="legal">This action is legal</option>
            <option value="illegal">This action is illegal; another is legal</option>
            <option value="none_legal">No actions are legal (allow all)</option>
          </select>
        </label>
      </div>
      <div className="policy-score-equation" aria-label="Final selection score calculation">
        <span><small>Value Q̄</small><b>{valueScore.toFixed(3)}</b></span><span aria-hidden="true">+</span>
        <span><small>Bonus U</small><b>{policyScore.toFixed(3)}</b></span><span aria-hidden="true">+</span>
        <span><small>Tie noise ε</small><b>{selectionNoise.toExponential(0)}</b></span><span aria-hidden="true">→</span>
        <span className="policy-score-result"><small>After legal mask</small><output aria-label="Final selection score" aria-live="polite">{scoreLabel}</output></span>
      </div>
      <p className="puct-muted">U = √N × c(N) × P(a) / (n + 1), where c(N) = 1.25 + ln((N + 19652 + 1) / 19652) = {coefficient.toFixed(3)} here.</p>
      <div className="puct-controls">
        <button type="button" disabled={nodeVisits === maxVisits} onClick={() => setState({ ...state, nodeVisits: nodeVisits + 1, actionVisits: actionVisits + 1 })}>Visit this action</button>
        <button type="button" disabled={nodeVisits === maxVisits} onClick={() => setState({ ...state, nodeVisits: nodeVisits + 1 })}>Visit another action</button>
      </div>
      <p className="puct-muted">Visiting this action increments both N and n; visiting another increments only N. The probability and normalized value stay fixed.</p>
      <label className="policy-score-range">Surface range
        <select aria-label="Surface range" value={maxVisits} onChange={(event) => {
          const limit = Number(event.currentTarget.value);
          const visits = Math.min(nodeVisits, limit);
          setMaxVisits(limit);
          setState({ ...state, nodeVisits: visits, actionVisits: Math.min(actionVisits, visits - 1) });
        }}>
          <option value="64">1–64 node visits</option><option value="256">1–256 node visits</option><option value="1000">1–1,000 node visits</option>
        </select>
      </label>
      <PolicyScoreSurface probability={probability} nodeVisits={nodeVisits} actionVisits={actionVisits} score={score} limit={maxVisits} normalizedValue={normalizedValue} legality={legality}
        onSelect={(node, action) => setState({ ...state, nodeVisits: node, actionVisits: action })} />
      <p className="policy-score-reading"><strong>Final score: {scoreLabel}</strong> · Value: {valueScore.toFixed(3)} · Bonus: {policyScore.toFixed(3)}.
        {' '}{legality === 'illegal' ? 'The legal-action mask excludes this action with −∞.' : legality === 'none_legal' ? 'With no legal actions, the fallback allows every action.' : 'Argmax compares this score with the other actions at the same node.'}</p>
      <figcaption>X: node visits. Y: action visits. Height and darker color: final selection score, Q̄ + U + ε after the legal-action mask. Probability and the visited-action normalized value are held fixed; Q̄ is zero at n = 0. The surface interpolates between visit counts. The illustrative noise draw is fixed at 5 × 10⁻⁸; training draws fresh noise. Scores of −∞ have no finite surface. Click the surface or use the sliders to select a point. The hatched region is impossible because N includes one initial prediction.</figcaption>
    </figure>
  );
}
