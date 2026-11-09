import { computePolicyScore } from './policyScore.ts';

export const actionNames = ['A', 'B', 'C', 'D'];
export const simulationCount = 8;

export type SearchNode = {
  id: number;
  parent: number;
  action: number;
  depth: number;
  visits: number;
  value: number;
  prediction: number;
  priors: number[];
  children: number[];
  counts: number[];
};
export type Score = { prior: number; visits: number; q: number; normalized: number; bonus: number; total: number };
export type Frame = {
  phase: 'prepare' | 'select' | 'expand' | 'backup' | 'output';
  simulation: number;
  nodes: SearchNode[];
  path: number[];
  parent: number;
  action: number;
  scores: Score[];
  returns: { node: number; value: number }[];
};

function makeNode(id: number, parent: number, action: number, depth: number, value: number): SearchNode {
  return {
    id, parent, action, depth, value, prediction: value, visits: 1,
    priors: depth === 0 ? [0.5, 0.3, 0.2, 0] : [0.45, 0.35, 0.2, 0],
    children: [-1, -1, -1, -1], counts: [0, 0, 0, 0],
  };
}

export function selectionScores(nodes: SearchNode[], node: SearchNode): Score[] {
  const q = node.children.map((child) => child < 0 ? node.value : -nodes[child].value);
  const minimum = Math.min(node.value, ...q);
  const maximum = Math.max(node.value, ...q);
  return node.priors.map((prior, action) => {
    const visits = node.counts[action];
    const normalized = visits === 0 ? 0 : (q[action] - minimum) / Math.max(maximum - minimum, 1e-8);
    const bonus = computePolicyScore(prior, node.visits, visits);
    return { prior, visits, q: visits === 0 ? 0 : q[action], normalized, bonus, total: action === 3 ? -Infinity : normalized + bonus };
  });
}

// A deterministic, nonterminal toy tree: exact PUCT arithmetic, illustrative network values.
// Reward is 0 and discount is -1 on every edge; no PRNG tie-breaking is needed here.
export function buildTrace(simulations: number = simulationCount): Frame[] {
  const nodes = [makeNode(0, -1, -1, 0, 0.1)];
  const frames: Frame[] = [];
  function record(phase: Frame['phase'], simulation: number, path: number[], parent: number, action: number, scores: Score[], returns: Frame['returns'] = []): void {
    frames.push({ phase, simulation, nodes: structuredClone(nodes), path: [...path], parent, action, scores, returns });
  }
  record('prepare', 0, [0], 0, -1, []);
  for (let simulation = 1; simulation <= simulations; simulation++) {
    const path = [0];
    let parent = 0;
    let action = 0;
    let scores: Score[];
    while (true) {
      scores = selectionScores(nodes, nodes[parent]);
      action = scores.reduce((best, score, index) => score.total > scores[best].total ? index : best, 0);
      const child = nodes[parent].children[action];
      if (child === -1) break;
      path.push(child);
      parent = child;
    }
    record('select', simulation, path, parent, action, scores);
    const depth = nodes[parent].depth + 1;
    const rootAction = path.length > 1 ? nodes[path[1]].action : action;
    const rootValue = [0.2, 0.65, -0.35][rootAction] + (depth > 1 ? (action - 1) * 0.15 : 0);
    const prediction = rootValue * (depth % 2 === 0 ? 1 : -1);
    const leaf = makeNode(nodes.length, parent, action, depth, prediction);
    nodes[parent].children[action] = leaf.id;
    nodes.push(leaf);
    path.push(leaf.id);
    record('expand', simulation, path, parent, action, scores);
    const returns = [{ node: leaf.id, value: prediction }];
    let value = prediction;
    for (let index = path.length - 2; index >= 0; index--) {
      const ancestor = nodes[path[index]];
      value = -value;
      ancestor.value = (ancestor.value * ancestor.visits + value) / (ancestor.visits + 1);
      ancestor.visits++;
      ancestor.counts[nodes[path[index + 1]].action]++;
      returns.push({ node: ancestor.id, value });
    }
    record('backup', simulation, path, parent, action, scores, returns);
  }
  record('output', simulations, [], 0, -1, []);
  return frames;
}

export function actionWeights(counts: number[], legal: boolean[]): number[] {
  const total = counts.reduce((sum, count) => sum + count, 0);
  if (total > 0) return counts.map((count) => count / total);
  const available = legal.some(Boolean) ? legal : legal.map(() => true);
  const size = available.filter(Boolean).length;
  return available.map((allowed) => allowed ? 1 / size : 0);
}

export function samplingWeights(weights: number[], temperature: number): number[] {
  const maximum = Math.max(...weights);
  const scaled = weights.map((weight) => temperature === 0 ? Number(weight === maximum) : (weight / maximum) ** (1 / temperature));
  const total = scaled.reduce((sum, value) => sum + value, 0);
  return scaled.map((value) => value / total);
}
