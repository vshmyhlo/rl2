import { useEffect, useRef, useState, type JSX } from 'react';
import { scaleLinear } from 'd3-scale';
import './VllmSamplerDemo.css';

type EventKind = 'model' | 'tool' | 'feedback';
type BlockKind = 'prompt' | 'sampled' | 'result';
type RolloutEvent = {
  kind: EventKind;
  start: number;
  end: number;
  label: string;
  text: string;
  appended: boolean;
};
type Rollout = {
  id: string;
  group: string;
  prompt: string;
  reward: number;
  stop: string;
  events: RolloutEvent[];
};
type TokenBlock = { kind: BlockKind; label: string; text: string };

const rollouts: Rollout[] = [
  {
    id: 'A0', group: 'A', prompt: 'Look up the parcel status and submit it.', reward: 1,
    stop: 'environment_done',
    events: [
      { kind: 'model', start: 0, end: 1, label: 'Turn 1', text: '<tool_call>{"name":"lookup","arguments":{"parcel":"42"}}</tool_call>', appended: true },
      { kind: 'tool', start: 1, end: 3, label: 'lookup', text: 'Status: delivered. Reward +0; done = false.', appended: true },
      { kind: 'model', start: 3, end: 5, label: 'Turn 2', text: '<tool_call>{"name":"submit","arguments":{"status":"delivered"}}</tool_call>', appended: true },
      { kind: 'tool', start: 5, end: 6, label: 'submit', text: 'Correct submission. Reward +1; done = true.', appended: false },
    ],
  },
  {
    id: 'A1', group: 'A', prompt: 'Look up the parcel status and submit it.', reward: 0,
    stop: 'agent_completed',
    events: [
      { kind: 'model', start: 0, end: 2, label: 'Turn 1', text: '<tool_call>{"name":"lookup","arguments":</tool_call>', appended: true },
      { kind: 'feedback', start: 2, end: 3, label: 'error', text: 'Invalid JSON: correct the tool-call arguments and retry. No environment step ran.', appended: true },
      { kind: 'model', start: 3, end: 6, label: 'Turn 2', text: 'I could not retrieve the parcel status.', appended: true },
    ],
  },
  {
    id: 'B0', group: 'B', prompt: 'Calculate 2 + 3 and submit the answer.', reward: 1,
    stop: 'environment_done',
    events: [
      { kind: 'model', start: 0, end: 1, label: 'Turn 1', text: '<tool_call>{"name":"submit","arguments":{"answer":5}}</tool_call>', appended: true },
      { kind: 'tool', start: 1, end: 2, label: 'submit', text: 'Correct answer. Reward +1; done = true.', appended: false },
    ],
  },
  {
    id: 'B1', group: 'B', prompt: 'Calculate 2 + 3 and submit the answer.', reward: 0,
    stop: 'agent_completed',
    events: [
      { kind: 'model', start: 0, end: 2, label: 'Turn 1', text: '<tool_call>{"name":"calculate","arguments":{"expression":"2+3"}}</tool_call>', appended: true },
      { kind: 'tool', start: 2, end: 3, label: 'calculate', text: 'Result: 5. Reward +0; done = false.', appended: true },
      { kind: 'model', start: 3, end: 4, label: 'Turn 2', text: 'The answer is 5.', appended: true },
    ],
  },
];

const stages = [
  ['Launch four rollouts', 'Two tasks × two independent attempts. Every rollout starts a generation request with its own prompt and environment.'],
  ['Generation hands off to tools', 'A0 waits for lookup; B0 submits an answer. A1 and B1 are still generating on the shared vLLM server.'],
  ['One rollout finishes', 'B0 is done, but its group waits for B1. A1 receives parsing feedback; B1 runs calculate.'],
  ['Tool results become context', 'A0, A1, and B1 append feedback and send their full histories back to vLLM for turn 2.'],
  ['Group B is ready', 'B1 stops without a tool call. Its text answer never invokes submit, so its reward stays zero. Group B can now be scored.'],
  ['The slow group continues', 'A0 executes submit while A1 is still generating. The batcher already has group B, but needs two more rollouts.'],
  ['Both groups are scored', 'Group A finishes. With rewards [1, 0] in each group and no length penalty, advantages are [+0.5, −0.5].'],
  ['Assemble the learner batch', 'Consume B0, B1, A0, A1 in ready-group order. Right-pad the transcripts; keep context visible and mask it out of the loss.'],
  ['Update, then publish when due', 'The learner uses sampled-token log-probabilities and rollout advantages for an update. A scheduled weight publication feeds future sampling.'],
];
const timelineScale = scaleLinear().domain([0, 6]).range([0, 100]);
const batchOrder = [2, 3, 0, 1];
const blockWidth = 6;

function finish(rollout: Rollout): number {
  return rollout.events[rollout.events.length - 1].end;
}

function blocks(rollout: Rollout, step: number): TokenBlock[] {
  return [
    { kind: 'prompt', label: 'P', text: 'System + user prompt + tool schemas + assistant prefix' } as TokenBlock,
    ...rollout.events.filter((event) => event.appended && event.end <= step).map((event): TokenBlock => ({
      kind: event.kind === 'model' ? 'sampled' : 'result',
      label: event.kind === 'model' ? `A${event.label.slice(-1)}` : 'O',
      text: event.kind === 'model' ? event.text : 'Tool result / error feedback + next assistant prefix',
    })),
  ];
}

export default function VllmSamplerDemo(): JSX.Element {
  const [step, setStep] = useState(0);
  const [selected, setSelected] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [showMask, setShowMask] = useState(false);
  const [stale, setStale] = useState(false);
  const [visible, setVisible] = useState(true);
  const [reducedMotion, setReducedMotion] = useState(false);
  const figure = useRef<HTMLElement>(null);
  const rollout = rollouts[selected];
  const transcript = blocks(rollout, step);
  const complete = step >= 7;
  const active = rollouts.flatMap((item) => item.events.filter((event) => event.start <= step && event.end > step));
  const generatedTurns = rollouts.reduce((count, item) => count + item.events.filter((event) => event.kind === 'model' && event.end <= step).length, 0);
  const readyGroups = ['A', 'B'].filter((group) => rollouts.filter((item) => item.group === group).every((item) => finish(item) <= step));

  useEffect(() => {
    const element = figure.current;
    if (!element) return;
    const observer = new IntersectionObserver((entries) => {
      // A resize can queue multiple changes for this target; use the latest one.
      setVisible(entries[entries.length - 1].isIntersecting);
    });
    observer.observe(element);
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const updatePreference = (): void => {
      setReducedMotion(preference.matches);
      if (preference.matches) setPlaying(false);
    };
    updatePreference();
    preference.addEventListener('change', updatePreference);
    return () => {
      observer.disconnect();
      preference.removeEventListener('change', updatePreference);
    };
  }, []);

  useEffect(() => {
    if (!playing || !visible || reducedMotion) return;
    const timeout = window.setTimeout(() => {
      if (step >= stages.length - 1) setPlaying(false);
      else {
        setStep(step + 1);
        if (step + 1 === stages.length - 1) setPlaying(false);
      }
    }, 1800);
    const pauseWhenHidden = (): void => { if (document.hidden) setPlaying(false); };
    document.addEventListener('visibilitychange', pauseWhenHidden);
    return () => {
      window.clearTimeout(timeout);
      document.removeEventListener('visibilitychange', pauseWhenHidden);
    };
  }, [playing, step, visible, reducedMotion]);

  function seek(next: number): void {
    setPlaying(false);
    setStep(next);
  }

  return (
    <figure ref={figure} className="vllm-demo">
      <div className="vs-heading"><strong>A batch in flight</strong><span>2 tasks · G = 2 · B = 4</span></div>
      <p className="vs-muted">Follow the schedule, then select a rollout to inspect its history.</p>
      <div className="vs-controls">
        <button type="button" className="vs-primary" disabled={reducedMotion} aria-pressed={playing} onClick={() => {
          if (step === stages.length - 1) setStep(0);
          setPlaying(!playing);
        }}>{playing ? 'Pause' : step === stages.length - 1 ? 'Replay' : 'Play'}</button>
        <button type="button" disabled={step === 0} onClick={() => seek(step - 1)}>Back</button>
        <button type="button" disabled={step === stages.length - 1} onClick={() => seek(step + 1)}>Next</button>
        <button type="button" onClick={() => { seek(0); setSelected(0); setStale(false); setShowMask(false); }}>Reset</button>
        <label className="vs-progress">Schedule <output>{step} / 8</output>
          <input aria-label="Sampling schedule" type="range" min="0" max="8" step="1" value={step}
            onChange={(event) => seek(event.currentTarget.valueAsNumber)} />
        </label>
      </div>
      {reducedMotion && <p className="vs-muted">Reduced motion is on. Use Next, Back, or the schedule slider to explore.</p>}
      <div className="vs-stage" role="status" aria-atomic="true">
        <strong>{String(step + 1).padStart(2, '0')} / {stages[step][0]}</strong>
        <p>{stages[step][1]}</p>
      </div>
      <ol className="vs-pipeline" aria-label="Sampling pipeline">
        <li className={step < 6 ? 'vs-current' : ''}>Actor<span>4 environments</span></li>
        <li className={step < 6 ? 'vs-current' : ''}>vLLM ↔ tools<span>Repeated turns</span></li>
        <li className={step >= 4 && step <= 7 ? 'vs-current' : ''}>Batcher<span>{readyGroups.length} / 2 groups ready</span></li>
        <li className={step === 8 ? 'vs-current' : ''}>Learner<span>{step === 8 ? 'One update' : 'Awaiting batch'}</span></li>
      </ol>
      <div className="vs-legend"><span><i className="vs-model" />Model response</span><span><i className="vs-tool" />Tool execution</span><span><i className="vs-feedback" />Error feedback</span></div>
      <div className="vs-scroll" role="region" aria-label="Concurrent rollout timeline" tabIndex={0}>
        <div className="vs-timeline">
          <div className="vs-time-heading"><span>Rollout</span><span>Illustrative time → (unequal turn lengths)</span><span>Return</span></div>
          {rollouts.map((item, index) => (
            <div className={`vs-lane ${selected === index ? 'vs-selected' : ''}`} key={item.id}>
              <button type="button" className="vs-row-label" aria-pressed={selected === index} onClick={() => setSelected(index)}>{item.id}<small>Task {item.group}</small></button>
              <div className="vs-track">
                {item.events.map((event, eventIndex) => (
                  <button type="button" key={eventIndex} className={`vs-event vs-${event.kind} ${event.start > step ? 'vs-future' : ''} ${event.start <= step && event.end > step ? 'vs-running' : ''}`}
                    style={{ left: `${timelineScale(event.start)}%`, width: `${timelineScale(event.end - event.start)}%` }}
                    aria-label={`${item.id}, ${event.label}: ${event.end <= step ? 'completed' : event.start <= step ? 'in progress' : 'upcoming'}`}
                    onClick={() => setSelected(index)}>{event.label}</button>
                ))}
                <span className="vs-cursor" style={{ left: `${timelineScale(Math.min(step, 6))}%` }} />
              </div>
              <span className="vs-return">{finish(item) <= step ? item.reward.toFixed(1) : '…'}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="vs-counters">
        <span><b>{active.filter((event) => event.kind === 'model').length}</b> model requests in flight</span>
        <span><b>{active.filter((event) => event.kind === 'tool').length}</b> tools in flight</span>
        <span><b>{generatedTurns}</b> completed turns</span>
      </div>

      <section className="vs-inspector" aria-label="Selected rollout">
        <div className="vs-heading"><strong>Inside rollout {rollout.id}</strong><span>{finish(rollout) <= step ? rollout.stop : 'In progress'}</span></div>
        <p className="vs-prompt"><b>Task {rollout.group}:</b> {rollout.prompt}</p>
        <ol className="vs-transcript">
          {rollout.events.filter((event) => event.start <= step).map((event, index) => (
            <li key={index}>
              <span className={`vs-badge vs-${event.kind}`}>{event.kind === 'model' ? event.label : event.kind === 'tool' ? 'Environment' : 'Parser'}</span>
              <div>{event.end > step ? <em>{event.kind === 'model' ? 'Generating response…' : event.kind === 'tool' ? 'Executing tool…' : 'Preparing error feedback…'}</em> : <>
                <code>{event.text}</code>
                <small>{event.kind === 'model' ? 'Append exact sampled tokens · loss mask = 1 · retain log-probs + weight versions' : event.appended ? 'Append observation as context · loss mask = 0' : 'Terminal result: reward recorded; no next-turn context appended'}</small>
              </>}</div>
            </li>
          ))}
        </ol>
        <div className="vs-token-strip" aria-label={`${rollout.id} appended transcript blocks`}>
          {transcript.map((block, index) => <span key={index} className={`vs-block vs-${block.kind}`} title={block.text}>{block.label}<small>mask {block.kind === 'sampled' ? '1' : '0'}</small></span>)}
        </div>
        <p className="vs-muted">P = initial context · A = assistant response · O = observation + next assistant prefix. Blocks appear when each response or tool result is returned.</p>
      </section>

      <section className="vs-batch" aria-label="Learner batch">
        <div className="vs-heading"><strong>The learner batch</strong><span>{complete ? '4 rows × T token positions' : `${readyGroups.length * 2} / 4 rows in ready groups`}</span></div>
        {!complete ? <p className="vs-muted">{readyGroups.length ? `Group ${readyGroups.join(' and ')} scored and ready. ` : 'Wait for complete groups. '}Advance to schedule step 7 to inspect the padded batch.</p> : <>
          <div className="vs-batch-controls">
            <label><input type="checkbox" checked={showMask} onChange={(event) => setShowMask(event.currentTarget.checked)} /> Show loss mask</label>
            <label><input type="checkbox" checked={stale} onChange={(event) => setStale(event.currentTarget.checked)} /> Inspect stale-token masking</label>
          </div>
          <p className="vs-muted">{stale ? 'What-if: all tokens were sampled at v0, consumed at v1 with max_staleness = 0. Their loss masks become zero; transcript tokens stay visible to attention.' : 'All sampled tokens are within the staleness limit. Every assistant block, including tool-call text, contributes eligible tokens.'}</p>
          <div className="vs-scroll" role="region" aria-label="Padded rollout batch" tabIndex={0}>
            <table className="vs-batch-table">
              <thead><tr><th scope="col">Row</th><th scope="col" colSpan={blockWidth}>Transcript → right padding</th><th scope="col">Adv.</th></tr></thead>
              <tbody>{batchOrder.map((index, row) => {
                const item = rollouts[index];
                const rowBlocks = blocks(item, 6);
                return <tr key={item.id} className={selected === index ? 'vs-selected' : ''}>
                  <th scope="row"><button type="button" aria-pressed={selected === index} onClick={() => setSelected(index)}>{row}: {item.id}</button></th>
                  {Array.from({ length: blockWidth }, (_, column) => {
                    const block = rowBlocks[column];
                    const mask = block?.kind === 'sampled' && !stale;
                    return <td key={column} className={`vs-block vs-${block?.kind ?? 'padding'} ${stale && block?.kind === 'sampled' ? 'vs-stale' : ''}`}
                      title={block ? `${block.text}; attention_mask = 1; loss_mask = ${mask ? 1 : 0}` : 'Right padding; attention_mask = 0; loss_mask = 0'}>
                      {showMask ? (mask ? '1' : '0') : block?.label ?? 'pad'}
                    </td>;
                  })}
                  <td className="vs-advantage">{item.reward === 1 ? '+0.5' : '−0.5'}</td>
                </tr>;
              })}</tbody>
            </table>
          </div>
          <p className="vs-muted">Each cell represents a variable-length token block, not one token. Gray hatching is padding; its attention mask and loss mask are both zero. Actual padding width follows the configured multiple.</p>
          <div className="vs-weight-loop">Learner update <span aria-hidden="true">→</span> publish weights when due <span aria-hidden="true">↩</span> vLLM sampling</div>
        </>}
      </section>
      <figcaption>Illustrative schedule, not a live run. G is attempts per task; B is rollouts per learner batch. One turn is one model response; it may trigger multiple tool calls. Playback starts only when you press Play.</figcaption>
    </figure>
  );
}
