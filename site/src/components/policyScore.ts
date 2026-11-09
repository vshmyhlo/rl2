/** Same exploration term as compute_policy_score in alphazero/train.py. */
export function explorationCoefficient(nodeVisits: number): number {
  return 1.25 + Math.log((nodeVisits + 19652 + 1) / 19652);
}

export function computePolicyScore(actionProbability: number, nodeVisits: number, actionVisits: number): number {
  return Math.sqrt(nodeVisits) * explorationCoefficient(nodeVisits) * actionProbability / (actionVisits + 1);
}
