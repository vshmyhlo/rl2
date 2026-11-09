import { useId, useMemo, useState, type JSX } from 'react';
import { policyDistributions, sampleLogGamma } from './dirichletNoise';
import './PuctDemo.css';
import './DirichletNoiseDemo.css';

const logits = [0, 0.3, 0.8, 0.4, 0.1, -0.2];
const actions = ['A', 'B', 'C', 'D', 'E', 'F'];
const legal = actions.map(() => true);

export default function DirichletNoiseDemo(): JSX.Element {
  const id = useId();
  const [alpha, setAlpha] = useState(0.3);
  const [fraction, setFraction] = useState(0.25);
  const [seed, setSeed] = useState(7);
  const logNoise = useMemo(() => sampleLogGamma(alpha, actions.length, seed), [alpha, seed]);
  const distributions = policyDistributions(logits, legal, logNoise, fraction);
  const panels = [
    { key: 'raw', title: 'Original', note: 'Fixed, slightly peaked policy · p' },
    { key: 'noise', title: 'Dirichlet', note: 'Random distribution · η' },
    { key: 'mixed', title: 'Mixed', note: `${(100 * (1 - fraction)).toFixed(0)}% original + ${(100 * fraction).toFixed(0)}% Dirichlet` },
  ] as const;

  return (
    <figure className="puct-demo dirichlet-demo" aria-labelledby={`${id}-title`}>
      <div className="puct-heading">
        <strong id={`${id}-title`}>Mixing a policy with Dirichlet noise</strong>
      </div>
      <div className="dirichlet-panels">
        {panels.map(panel => (
          <section key={panel.key} className={`dirichlet-panel dirichlet-${panel.key}`} aria-labelledby={`${id}-${panel.key}`}>
            <h3 id={`${id}-${panel.key}`}>{panel.title}</h3>
            <p>{panel.note}</p>
            <ul>
              {distributions[panel.key].map((probability, index) => (
                <li key={actions[index]}>
                  <output aria-label={`${panel.title}, action ${actions[index]}`}>{(probability * 100).toFixed(1)}%</output>
                  <span className="dirichlet-bar-track" aria-hidden="true"><span style={{ height: `${probability * 100}%` }} /></span>
                  <span>{actions[index]}</span>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
      <div className="dirichlet-parameters">
        <label htmlFor={`${id}-alpha`}>Concentration α <output>{alpha.toFixed(2)}</output>
          <input id={`${id}-alpha`} type="range" min="0.05" max="3" step="0.05" value={alpha}
            onChange={event => setAlpha(event.currentTarget.valueAsNumber)} />
          <small>Smaller α tends to make sparse draws; larger α tends toward equal shares.</small>
        </label>
        <label htmlFor={`${id}-fraction`}>Noise fraction ε <output>{fraction.toFixed(2)}</output>
          <input id={`${id}-fraction`} type="range" min="0" max="1" step="0.01" value={fraction}
            onChange={event => setFraction(event.currentTarget.valueAsNumber)} />
          <small>0 keeps the original; 1 uses only the Dirichlet sample.</small>
        </label>
      </div>
      <div className="puct-controls">
        <button type="button" className="puct-primary" onClick={() => setSeed(current =>
          (current + 1 + Math.floor(Math.random() * 4294967295)) >>> 0)}>Sample new Dirichlet</button>
        <span className="puct-muted">Changing ε keeps the Dirichlet sample fixed.</span>
      </div>
      <figcaption>Each bar is the probability of one discrete action, A–F, on a shared 0–100% height scale. Mixed = (1 − ε) × original + ε × Dirichlet. Noise is sampled directly in the browser by normalizing Gamma draws. Changing α recomputes the draw with the same seed; individual samples can jump, including near α = 1. Samples follow the same distribution as Python, but not JAX’s exact random sequence.</figcaption>
    </figure>
  );
}
