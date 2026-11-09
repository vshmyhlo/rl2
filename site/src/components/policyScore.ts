/** Same exploration term as compute_policy_score in alphazero/train.py. */
export function explorationCoefficient(nodeVisits: number): number {
  return 1.25 + Math.log((nodeVisits + 19652 + 1) / 19652);
}

export function computePolicyScore(actionProbability: number, nodeVisits: number, actionVisits: number): number {
  return Math.sqrt(nodeVisits) * explorationCoefficient(nodeVisits) * actionProbability / (actionVisits + 1);
}

// Hold one illustrative uniform draw fixed so moving the controls is reproducible.
export const selectionNoise = 1e-7 * 0.5;
export type Legality = 'legal' | 'illegal' | 'none_legal';

/** Final score before argmax; normalizedValue holds the parent/sibling normalization fixed. */
export function computeSelectionScore(
  actionProbability: number, nodeVisits: number, actionVisits: number,
  normalizedValue: number, legality: Legality,
): number {
  if (legality === 'illegal') return -Infinity;
  const valueScore = actionVisits > 0 ? normalizedValue : 0;
  return valueScore + computePolicyScore(actionProbability, nodeVisits, actionVisits) + selectionNoise;
}
