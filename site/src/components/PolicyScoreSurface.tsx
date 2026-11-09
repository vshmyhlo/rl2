import { useId, type JSX } from 'react';
import { scaleLinear } from 'd3-scale';
import { computePolicyScore } from './policyScore';

type Point = { nodeVisits: number; actionVisits: number; score: number };
type Props = Point & {
  probability: number;
  limit: number;
  onSelect: (nodeVisits: number, actionVisits: number) => void;
};

export default function PolicyScoreSurface({ probability, nodeVisits, actionVisits, score, limit, onSelect }: Props): JSX.Element {
  const id = useId();
  const ceiling = computePolicyScore(1, limit, 0);
  const color = scaleLinear<string>().domain([0, ceiling]).range(['#e7eaf5', '#485689']);
  const project = (node: number, action: number, bonus: number): [number, number] => {
    const x = (node - 1) / (limit - 1);
    const y = action / (limit - 1);
    return [80 + 300 * x + 170 * y, 300 - 100 * x + 90 * y - 170 * bonus / ceiling];
  };
  const point = (node: number, action: number): Point => ({
    nodeVisits: node, actionVisits: action, score: computePolicyScore(probability, node, action),
  });
  const coordinates = (points: Point[]): string => points.map(p => project(p.nodeVisits, p.actionVisits, p.score).join(',')).join(' ');
  // Denser samples near zero capture the steep 1 / (actionVisits + 1) ridge.
  const grid = Array.from({ length: 33 }, (_, i) => (limit - 1) * (i / 32) ** 2);
  const faces: { points: Point[]; depth: number }[] = [];
  for (let x = 0; x < grid.length - 1; x++) {
    for (let y = 0; y <= x; y++) {
      const points = [point(1 + grid[x], grid[y]), point(1 + grid[x + 1], grid[y]), point(1 + grid[x + 1], grid[y + 1])];
      if (y < x) points.push(point(1 + grid[x], grid[y + 1]));
      faces.push({ points, depth: grid[x] + grid[x + 1] - grid[y] - grid[y + 1] });
    }
  }
  faces.sort((a, b) => b.depth - a.depth);
  const [dotX, dotY] = project(nodeVisits, actionVisits, score);
  const [floorX, floorY] = project(nodeVisits, actionVisits, 0);
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return <>
    <p className="policy-score-scroll-hint">Swipe or scroll horizontally to see both axes.</p>
    <div className="policy-score-chart" role="region" aria-label="Exploration bonus surface, scroll horizontally on small screens" tabIndex={0}>
      <svg viewBox="0 0 620 445" role="img" aria-label={`3D surface: X is node visits, Y is action visits, height and color are exploration bonus. Probability ${probability.toFixed(2)}. Selected N=${nodeVisits}, n=${actionVisits}, bonus=${score.toFixed(3)}.`}>
        <defs>
          <pattern id={`${id}-invalid`} width="7" height="7" patternUnits="userSpaceOnUse"><path d="M0 7L7 0" stroke="#deded8" strokeWidth="1" /></pattern>
        </defs>
        <polygon points="80,300 380,200 550,290 250,390" fill="#f6f5f0" />
        <polygon points="80,300 250,390 550,290" fill={`url(#${id}-invalid)`} />
        {ticks.map(t => {
          const n = Math.round(t * (limit - 1));
          const a = project(1 + n, 0, 0);
          const b = project(1 + n, limit - 1, 0);
          const c = project(1, n, 0);
          const d = project(limit, n, 0);
          return <g key={t} className="policy-score-grid">
            <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} />
            <line x1={c[0]} y1={c[1]} x2={d[0]} y2={d[1]} />
          </g>;
        })}
        <g className="policy-score-surface-faces">{faces.map(({ points }, index) => {
          const node = Math.round(points.reduce((sum, p) => sum + p.nodeVisits, 0) / points.length);
          const action = Math.min(node - 1, Math.round(points.reduce((sum, p) => sum + p.actionVisits, 0) / points.length));
          const bonus = points.reduce((sum, p) => sum + p.score, 0) / points.length;
          return <polygon key={index} points={coordinates(points)} fill={color(bonus)} onClick={() => onSelect(node, action)}>
            <title>{`Select N=${node}, n=${action}: bonus ${computePolicyScore(probability, node, action).toFixed(3)}`}</title>
          </polygon>;
        })}</g>
        <path className="policy-score-axis" d="M80 125V300L250 390L550 290" />
        {ticks.map(t => {
          const n = Math.round(t * (limit - 1));
          const x = project(1 + n, limit - 1, 0);
          const y = project(1, n, 0);
          return <g key={t}>
            <text x={x[0] + 8} y={x[1] + 17} textAnchor="middle">{Math.round(1 + n)}</text>
            <text x={y[0] - 13} y={y[1] + 12} textAnchor="end">{Math.round(n)}</text>
            <text x="69" y={304 - 170 * t} textAnchor="end">{(ceiling * t).toFixed(1)}</text>
          </g>;
        })}
        <text x="15" y="108">Bonus U (height)</text>
        <text x="410" y="393" textAnchor="middle">X · Node visits N</text>
        <text x="113" y="385" textAnchor="middle">Y · Action visits n</text>
        <text x="292" y="346" textAnchor="middle" className="policy-score-invalid-label">n &gt; N − 1: not reachable</text>
        <line className="policy-score-guide" x1={dotX} x2={floorX} y1={dotY} y2={floorY} />
        <circle className="policy-score-point" cx={dotX} cy={dotY} r="5" />
        <text x="310" y="432" textAnchor="middle">Orange dot: N = {nodeVisits}, n = {actionVisits}, U = {score.toFixed(3)}</text>
      </svg>
    </div>
    <div className="policy-score-color-key" aria-label={`Color scale: zero to ${ceiling.toFixed(2)} exploration bonus`}>
      <span>0</span><span className="policy-score-color-ramp" /><span>{ceiling.toFixed(2)} bonus</span>
    </div>
  </>;
}
