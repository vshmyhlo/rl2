import { useEffect, useRef, useState, type JSX } from 'react';
import { actionNames, actionWeights, buildTrace, samplingWeights, simulationCount, type Frame } from './puctTrace';
import './PuctDemo.css';

const frames = buildTrace();
const finalNodes = frames[frames.length - 1].nodes;
const positions = new Map<number, number>();
let slot = 0;
function place(id: number): number {
  const children = finalNodes[id].children.filter((child) => child >= 0);
  const xs = children.map(place);
  if (id === 0) xs.push(420, 540);
  const x = xs.length ? xs.reduce((sum, value) => sum + value, 0) / xs.length : 60 + slot++ * 120;
  positions.set(id, x);
  return x;
}
place(0);
const y = (depth: number): number => 38 + depth * 82;
const signed = (value: number): string => `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(2)}`;
const percent = (value: number): string => `${(value * 100).toFixed(value === 0 || value === 1 ? 0 : 1)}%`;
const phases = ['prepare', 'select', 'expand', 'backup', 'output'] as const;
const titles = { prepare: 'Prepare the root', select: 'Select a path', expand: 'Expand one leaf', backup: 'Back up the value', output: 'Return a policy and an action' };

function explanation(frame: Frame): string {
  const leaf = frame.nodes[frame.nodes.length - 1];
  switch (frame.phase) {
    case 'prepare': return 'The caller supplies the root state, value and policy logits. Mask invalid action D, allocate the tree, and count the root prediction as visit 1. No search edges have visits yet.';
    case 'select': return `Start at the root and follow the highest PUCT score until an edge has no child. Path: ${frame.path.map((id) => `n${id}`).join(' → ')} → action ${actionNames[frame.action]}. The table shows the last decision on this path.`;
    case 'expand': return `recurrent_fn steps a copy of the state and predicts the child policy and value. Store n${leaf.id} with value ${signed(leaf.prediction)} from its player's perspective and one node visit. Edge counts have not changed yet.`;
    case 'backup': return `Walk back to the root. Here reward = 0 and discount = −1, so the return changes sign at each edge. Update each ancestor's running mean and increment each traversed edge's visit count.`;
    case 'output': return `After ${simulationCount} simulations, normalize the root edge visits into action_weights. Temperature changes the distribution used to sample action, while action_weights stays the same. The complete search_tree is also returned.`;
  }
}

export default function PuctDemo(): JSX.Element {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [visible, setVisible] = useState(false);
  const [pageVisible, setPageVisible] = useState(true);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [temperature, setTemperature] = useState(1);
  const [speed, setSpeed] = useState(1);
  const [sample, setSample] = useState<number | null>(null);
  const figure = useRef<HTMLElement>(null);
  const frame = frames[step];
  const complete = step === frames.length - 1;
  const root = frame.nodes[0];
  const weights = actionWeights(root.counts, [true, true, true, false]);
  const sampling = samplingWeights(weights, temperature);
  const completedSimulations = root.counts.reduce((sum, count) => sum + count, 0);
  const ghost = frame.phase === 'select' ? finalNodes.find((node) => node.parent === frame.parent && node.action === frame.action) : undefined;

  useEffect(() => {
    const element = figure.current;
    if (!element) return;
    const observer = new IntersectionObserver((entries) => setVisible(entries[entries.length - 1].isIntersecting));
    observer.observe(element);
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const updatePreference = (): void => {
      setReducedMotion(preference.matches);
      if (preference.matches) setPlaying(false);
    };
    const updateVisibility = (): void => setPageVisible(!document.hidden);
    updatePreference();
    updateVisibility();
    preference.addEventListener('change', updatePreference);
    document.addEventListener('visibilitychange', updateVisibility);
    return () => {
      observer.disconnect();
      preference.removeEventListener('change', updatePreference);
      document.removeEventListener('visibilitychange', updateVisibility);
    };
  }, []);

  useEffect(() => {
    if (!playing || !visible || !pageVisible || reducedMotion) return;
    const timeout = window.setTimeout(() => {
      setStep((current) => Math.min(current + 1, frames.length - 1));
      if (step + 1 === frames.length - 1) setPlaying(false);
    }, 1800 / speed);
    return () => window.clearTimeout(timeout);
  }, [playing, visible, pageVisible, reducedMotion, step, speed]);

  function seek(next: number): void {
    setPlaying(false);
    setStep(next);
    setSample(null);
  }

  function drawAction(): void {
    const draw = Math.random();
    let cumulative = 0;
    const action = sampling.findIndex((probability) => {
      cumulative += probability;
      return draw < cumulative;
    });
    setSample(action);
  }

  return (
    <figure className="puct-demo" ref={figure}>
      <div className="puct-heading"><strong>One move, many imagined futures</strong><span>1 of B games · 8 simulations</span></div>
      <p className="puct-muted">Follow one search tree. A–C are legal actions; D is masked. The real game stays at the root while search explores copies.</p>
      <div className="puct-controls">
        <button className="puct-primary" type="button" disabled={reducedMotion} aria-pressed={playing} onClick={() => {
          if (complete) { setStep(0); setSample(null); }
          setPlaying(!playing);
        }}>{playing ? 'Pause' : complete ? 'Replay' : 'Play'}</button>
        <button type="button" disabled={step === 0} onClick={() => seek(step - 1)}>Back</button>
        <button type="button" disabled={complete} onClick={() => seek(step + 1)}>Next</button>
        <button type="button" onClick={() => { seek(0); setTemperature(1); setSpeed(1); }}>Reset</button>
        <label>Speed <select value={speed} onChange={(event) => setSpeed(Number(event.currentTarget.value))}><option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option></select></label>
      </div>
      <label className="puct-progress">Search progress <output>{completedSimulations} / {simulationCount} simulations backed up</output>
        <input type="range" aria-label="Search progress" min="0" max={frames.length - 1} step="1" value={step} onChange={(event) => seek(event.currentTarget.valueAsNumber)} />
      </label>
      {reducedMotion && <p className="puct-muted">Reduced motion is on. Use Next, Back, or the slider to explore.</p>}
      <ol className="puct-phases" aria-label="Search phases">{phases.map((phase, index) => <li key={phase} aria-current={frame.phase === phase ? 'step' : undefined}><span>{index + 1}</span>{phase === 'prepare' ? 'Prepare' : phase === 'output' ? 'Return' : phase[0].toUpperCase() + phase.slice(1)}</li>)}</ol>
      <div className="puct-stage" role="status" aria-atomic="true">
        <strong>{titles[frame.phase]}{['select', 'expand', 'backup'].includes(frame.phase) && ` · simulation ${frame.simulation}`}</strong>
        <p>{explanation(frame)}</p>
      </div>
      <div className="puct-tree-scroll" tabIndex={0} role="region" aria-label="Search tree diagram, scroll horizontally on small screens">
        <svg className="puct-tree" viewBox="0 0 600 408" role="img" aria-label={`${frame.nodes.length} stored nodes. Highlighted path: ${frame.path.map((id) => `n${id}`).join(', ') || 'none'}. Node values are from each node's player-to-move perspective.`}>
          <text className="puct-tree-note" x="12" y="17">Node ID · visits N · mean value V</text>
          {frame.nodes.filter((node) => node.id !== 0).map((node) => {
            const parent = frame.nodes[node.parent];
            const active = frame.path.includes(node.id);
            const x = positions.get(node.id)!;
            const px = positions.get(parent.id)!;
            return <g key={`edge-${node.id}`} className={active ? `puct-edge puct-active puct-${frame.phase}` : 'puct-edge'}>
              <path d={`M${px} ${y(parent.depth) + 25} L${x} ${y(node.depth) - 22}`} pathLength="1" />
              <text x={(px + x) / 2 + 9} y={(y(parent.depth) + y(node.depth)) / 2 + 4}>{actionNames[node.action]} · {parent.counts[node.action]}</text>
            </g>;
          })}
          {[2, 3].map((action) => <g key={action} className="puct-ghost">
            <path d={`M${positions.get(0)} ${y(0) + 25} L${action === 2 ? 420 : 540} ${y(1) - 22}`} />
            <text x={action === 2 ? 420 : 540} y={y(1)} textAnchor="middle">{actionNames[action]} · {action === 3 ? 'masked' : 'unvisited'}</text>
          </g>)}
          {ghost && <g className="puct-ghost puct-missing">
            <path d={`M${positions.get(frame.parent)} ${y(ghost.depth - 1) + 25} L${positions.get(ghost.id)} ${y(ghost.depth) - 22}`} />
            <rect x={positions.get(ghost.id)! - 48} y={y(ghost.depth) - 22} width="96" height="48" rx="8" />
            <text x={positions.get(ghost.id)} y={y(ghost.depth) + 5} textAnchor="middle">{actionNames[frame.action]} → new leaf</text>
          </g>}
          {frame.nodes.map((node) => <g key={node.id} className={`puct-node ${frame.path.includes(node.id) ? 'puct-active' : ''} ${frame.phase === 'expand' && node.id === frame.nodes.length - 1 ? 'puct-new' : ''}`} transform={`translate(${positions.get(node.id)}, ${y(node.depth)})`}>
            <rect x="-48" y="-22" width="96" height="48" rx="8" />
            <text textAnchor="middle" y="-3">{node.id === 0 ? 'n0 · root' : `n${node.id} · P${node.depth % 2}`}</text>
            <text textAnchor="middle" y="15">N {node.visits} · V {signed(node.value)}</text>
          </g>)}
        </svg>
      </div>
      <p className="puct-legend"><span>Orange outline = current path</span><span>Dashed edge = no stored child</span><span>Edge label = action · visits</span></p>
      <div className="puct-detail">
        {frame.phase === 'prepare' || frame.phase === 'output' ? <>
          <strong>{frame.phase === 'prepare' ? 'What enters the function' : 'What leaves the function'}</strong>
          <p>{frame.phase === 'prepare' ? 'root: prior_logits (B, A), value (B,), env_states (B, …). Here the legal priors are [0.50, 0.30, 0.20]. Self-play has already mixed in root noise.' : 'action (B,) is sampled once per game. action_weights (B, A) supplies the policy target. search_tree has 1 + num_simulations node slots per game.'}</p>
        </> : frame.phase === 'backup' ? <>
          <strong>Return flowing from leaf to root</strong>
          <div className="puct-backup">{frame.returns.map((item, index) => <span key={item.node}>{index > 0 && <b aria-hidden="true"> → </b>}n{item.node}: <b>{signed(item.value)}</b></span>)}</div>
          <code>new_mean = (old_mean × old_visits + return) / (old_visits + 1)</code>
          <p>Each node also counts its initial prediction. Root node visits = {root.visits}; root edge visits sum to {completedSimulations}.</p>
        </> : <>
          <strong>{frame.phase === 'select' ? `PUCT decision at n${frame.parent}` : `Scores used at n${frame.parent} before expansion`}</strong>
          <div className="puct-table-scroll" tabIndex={0} role="region" aria-label="Action selection scores">
            <table><thead><tr><th scope="col">Action</th><th scope="col">Prior P</th><th scope="col">Visits</th><th scope="col">Q̄</th><th scope="col">Bonus U</th><th scope="col">Q̄ + U</th></tr></thead>
              <tbody>{frame.scores.map((score, action) => <tr key={action} className={action === frame.action ? 'puct-chosen' : ''}><th scope="row">{actionNames[action]}{action === frame.action ? ' ←' : ''}</th><td>{score.prior.toFixed(2)}</td><td>{score.visits}</td><td>{score.normalized.toFixed(2)}</td><td>{score.bonus.toFixed(2)}</td><td>{action === 3 ? 'masked' : score.total.toFixed(2)}</td></tr>)}</tbody>
            </table>
          </div>
          <p>Q̄ rescales values using the parent and visited siblings; unvisited actions start at 0. U = c(N) × P × √N / (1 + edge visits). The arrow marks the largest score.</p>
        </>}
      </div>
      <section className="puct-output" aria-label="Visit counts and action sampling">
        <div className="puct-heading"><strong>{complete ? 'Search result' : 'Root policy so far'}</strong><span>{completedSimulations === 0 ? 'No visits: uniform legal fallback' : `Visits: [${root.counts.join(', ')}]`}</span></div>
        <div className="puct-distributions">{[
          { title: 'Prior P', values: root.priors, caption: 'Guides exploration' },
          { title: 'action_weights', values: weights, caption: 'Normalized visits · training target' },
          { title: 'Sampling probability', values: sampling, caption: `Temperature τ = ${temperature.toFixed(1)}` },
        ].map((distribution) => <div key={distribution.title}>
          <strong>{distribution.title}</strong>
          {distribution.values.map((value, action) => <div className="puct-bar-row" key={action} aria-label={`${distribution.title}, action ${actionNames[action]}: ${percent(value)}`}><span>{actionNames[action]}</span><span className="puct-bar-track"><span style={{ width: `${value * 100}%` }} /></span><output>{percent(value)}</output></div>)}
          <small>{distribution.caption}</small>
        </div>)}</div>
        <label className="puct-temperature">Sampling temperature <output>τ = {temperature.toFixed(1)}</output>
          <input type="range" min="0" max="2" step="0.1" value={temperature} aria-label="Sampling temperature" onChange={(event) => { setTemperature(event.currentTarget.valueAsNumber); setSample(null); }} />
        </label>
        <p className="puct-muted">{temperature === 0 ? 'At τ = 0, sample among the most-visited actions (ties remain random).' : 'Sampling probability ∝ action_weights^(1 / τ). Lower temperatures concentrate on the most-visited actions.'} The middle bars do not change.</p>
        <div className="puct-sample"><button type="button" disabled={!complete} onClick={drawAction}>Sample action</button><span role="status">{!complete ? 'Finish the search to draw an action.' : sample === null ? 'Ready to draw from the right-hand bars.' : `Sampled ${actionNames[sample]} · collect_selfplay now takes this action in the real game.`}</span></div>
      </section>
      <figcaption>Illustrative network predictions; PUCT selection, running means, and visit counts follow the local implementation. This example has only ongoing states (reward 0, discount −1) and no score ties. Terminal transitions instead use discount 0 and value 0. Each simulation expands one node per game; batched games may follow paths of different depths.</figcaption>
    </figure>
  );
}
