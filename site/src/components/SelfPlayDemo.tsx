import { Fragment, useState, type JSX } from 'react';
import './SelfPlayDemo.css';

const moves = 6;
const times = Array.from({ length: moves }, (_, t) => t);
const games = [
  { length: 3, firstPlayer: 0, winner: 0, outcome: 'P0 wins after 3 moves' },
  { length: 5, firstPlayer: 1, winner: 1, outcome: 'P1 wins after 5 moves' },
  { length: 6, firstPlayer: 0, winner: null, outcome: 'Move-limit draw' },
];

export default function SelfPlayDemo(): JSX.Element {
  const [collected, setCollected] = useState(moves);
  const [selected, setSelected] = useState({ t: 2, b: 0 });
  const { t, b } = selected;
  const game = games[b];
  const recorded = t < collected;
  const valid = t < game.length;
  const player = (game.firstPlayer + t) % 2;
  const complete = collected === moves;
  const value = !valid || game.winner === null ? 0 : player === game.winner ? 1 : -1;
  const positions = games.reduce((total, item) => total + Math.min(collected, item.length), 0);

  return (
    <figure className="selfplay-demo">
      <div className="az-heading">
        <strong>The self-play batch</strong>
        <span>B = 3 · T = 6</span>
      </div>
      <p className="az-intro">Games × time: rows are games, columns are scan steps. This (B, T) view swaps the scan's first two axes. Select a cell to look inside.</p>
      <div className="az-controls">
        <button type="button" onClick={() => setCollected(0)}>Reset</button>
        <button type="button" disabled={complete} onClick={() => setCollected((count) => count + 1)}>Next move</button>
        <label>
          <span>Scan progress <output>{collected} / {moves}</output></span>
          <input aria-label="Collected moves" type="range" min="0" max={moves} step="1" value={collected}
            onChange={(event) => setCollected(event.currentTarget.valueAsNumber)} />
        </label>
      </div>
      <p className="az-status" role="status">
        {collected === 0 ? 'Fresh games. No positions recorded yet.' :
          `${collected} time columns recorded · ${positions} real positions · ${collected * games.length - positions} padding slots`}
        {complete && ' · Outcome targets attached'}
      </p>
      <div className="az-grid-scroll" role="region" aria-label="Games by time batch" tabIndex={0}>
        <div className="az-grid">
          <span className="az-axis">b ↓ / t →</span>
          {times.map((time) => <strong className="az-axis" key={time}>t = {time}</strong>)}
          {games.map((item, index) => (
            <Fragment key={index}>
              <strong className="az-axis">Game {index}</strong>
              {times.map((time) => {
                const pending = time >= collected;
                const padding = time >= item.length;
                const actor = (item.firstPlayer + time) % 2;
                const active = time === t && index === b;
                return (
                  <button type="button" key={time} aria-pressed={active}
                    aria-label={`Game ${index}, time ${time}: ${pending ? 'not recorded' : padding ? 'padding' : `player ${actor}`}`}
                    className={`az-cell ${pending ? 'az-pending' : padding ? 'az-padding' : `az-game-${index}`}`}
                    onClick={() => setSelected({ t: time, b: index })}>
                    <strong>{pending ? '—' : padding ? 'pad' : `P${actor}`}</strong>
                    <small>{pending ? 'waiting' : padding ? 'masked' : time === item.length - 1 ? 'last move' : 'position'}</small>
                  </button>
                );
              })}
              <span className="az-outcome">{collected >= item.length ? item.outcome : 'In progress'}</span>
            </Fragment>
          ))}
        </div>
      </div>
      <div className="az-inspector" aria-live="polite" aria-atomic="true">
        <strong>Inside [b = {b}, t = {t}]</strong>
        {!recorded ? <p>This time column has not been collected yet. Advance the slider to record it.</p> : (
          <>
            <dl>
              <dt>observations</dt><dd>{valid ? `o[${b}, ${t}] · shape (*O)` : 'Preserved final observation'}</dd>
              <dt>policy_targets</dt><dd>{valid ? `π[${b}, ${t}] · shape (A,)` : 'Ignored search weights'}</dd>
              <dt>player_ids</dt><dd>{valid ? `P${player}` : 'Preserved final player ID'}</dd>
              <dt>policy_mask</dt><dd>{String(valid)}</dd>
              <dt>value_targets</dt><dd>{complete ? (value > 0 ? '+1' : String(value)) : 'Attached after the scan'}</dd>
              <dt>value_mask</dt><dd>{complete ? String(valid) : 'Attached after the scan'}</dd>
            </dl>
            <p>{!valid ? 'Padding: this game already ended. Neither head trains on this cell.' :
              !complete ? 'The observation and policy are recorded before taking this move.' :
              game.winner === null ? 'A move-limit draw: zero is a real value target, with both masks true.' :
              `P${game.winner} won. This position belongs to P${player}, so its value target is ${value > 0 ? '+1' : '−1'}.`}</p>
          </>
        )}
      </div>
      <div className="az-flat">
        <strong>Position indices for sampling</strong>
        <p>Read each game row left to right. The selected cell lands at flat index <b>{b * moves + t}</b>.</p>
        <div className="az-flat-games">
          {games.map((item, index) => (
            <div key={index}>
              <span>Game {index}</span>
              <div className="az-flat-slots">
                {times.map((time) => (
                  <span key={time}
                    className={`${time >= collected ? 'az-pending' : time >= item.length ? 'az-padding' : `az-game-${index}`} ${time === t && index === b ? 'az-selected' : ''}`}
                    title={`Flat index ${index * moves + time}: [b=${index}, t=${time}]`}>
                    {index * moves + time}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
      <figcaption>Illustrative games, with symbolic observations o and MCTS policies π. P0 / P1 are player IDs.
        Each cell stands for aligned entries across the recorded arrays; blank cells are unrevealed scan steps.
        The final batch has 18 slots: 14 real positions and 4 padding slots.</figcaption>
    </figure>
  );
}
