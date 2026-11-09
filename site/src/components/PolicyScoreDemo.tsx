import { useId, useState, type JSX } from 'react';
import { computePolicyScore, explorationCoefficient } from './policyScore';
import PolicyScoreSurface from './PolicyScoreSurface';
import './PuctDemo.css';
import './PolicyScoreDemo.css';

const initial = { probability: 0.6, nodeVisits: 32, actionVisits: 8 };

export default function PolicyScoreDemo(): JSX.Element {
  const id = useId();
  const [state, setState] = useState(initial);
  const [maxVisits, setMaxVisits] = useState(64);
  const { probability, nodeVisits, actionVisits } = state;
  const coefficient = explorationCoefficient(nodeVisits);
  const score = computePolicyScore(probability, nodeVisits, actionVisits);
  const unvisitedScore = computePolicyScore(probability, nodeVisits, 0);

  return (
    <figure className="puct-demo policy-score-demo" aria-labelledby={`${id}-title`}>
      <div className="puct-heading">
        <strong id={`${id}-title`}>What makes an action worth exploring?</strong>
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
      </div>
      <div className="policy-score-equation" aria-label="Exploration bonus calculation">
        <span><small>√N</small><b>{Math.sqrt(nodeVisits).toFixed(3)}</b></span><span aria-hidden="true">×</span>
        <span><small>c(N)</small><b>{coefficient.toFixed(3)}</b></span><span aria-hidden="true">×</span>
        <span><small>P(a)</small><b>{probability.toFixed(2)}</b></span><span aria-hidden="true">÷</span>
        <span><small>n + 1</small><b>{actionVisits + 1}</b></span><span aria-hidden="true">=</span>
        <span className="policy-score-result"><small>Bonus U</small><output aria-label="Exploration bonus" aria-live="polite">{score.toFixed(3)}</output></span>
      </div>
      <p className="puct-muted">c(N) = 1.25 + ln((N + 19652 + 1) / 19652). At this node it is {coefficient.toFixed(3)}.</p>
      <div className="puct-controls">
        <button type="button" disabled={nodeVisits === maxVisits} onClick={() => setState({ ...state, nodeVisits: nodeVisits + 1, actionVisits: actionVisits + 1 })}>Visit this action</button>
        <button type="button" disabled={nodeVisits === maxVisits} onClick={() => setState({ ...state, nodeVisits: nodeVisits + 1 })}>Visit another action</button>
      </div>
      <p className="puct-muted">Visiting this action increments both N and n; visiting another increments only N. The probability stays fixed.</p>
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
      <PolicyScoreSurface probability={probability} nodeVisits={nodeVisits} actionVisits={actionVisits} score={score} limit={maxVisits}
        onSelect={(node, action) => setState({ ...state, nodeVisits: node, actionVisits: action })} />
      <p className="policy-score-reading"><strong>Current bonus: {score.toFixed(3)}</strong> · Unvisited bonus: {unvisitedScore.toFixed(3)}.
        {' '}{probability === 0 ? 'A zero probability gives zero exploration bonus.' : `The visit denominator makes this ${(100 / (actionVisits + 1)).toFixed(1)}% of the unvisited bonus.`}</p>
      <figcaption>X: node visits. Y: action visits. Height and darker color: larger exploration bonus. Probability is held fixed across the surface; changing it raises or lowers the surface on a fixed height scale. Click the surface or use the sliders to select a point. The hatched region is impossible because N includes one initial prediction. Action selection also adds the normalized value score.</figcaption>
    </figure>
  );
}
