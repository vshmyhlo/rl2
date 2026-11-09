function randomSource(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return (state + 0.5) / 4294967296;
  };
}

// Marsaglia–Tsang sampling, retaining small Gamma draws in log space.
function logGamma(alpha: number, random: () => number): number {
  if (alpha < 1) return logGamma(alpha + 1, random) + Math.log(random()) / alpha;
  const d = alpha - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    const x = Math.sqrt(-2 * Math.log(random())) * Math.cos(2 * Math.PI * random());
    const base = 1 + c * x;
    if (base <= 0) continue;
    const v = base ** 3;
    const u = random();
    if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < x * x / 2 + d * (1 - v + Math.log(v))) {
      return Math.log(d) + 3 * Math.log(base);
    }
  }
}

/** Seeded for stable slider comparisons and server/client rendering, not JAX parity. */
export function sampleLogGamma(alpha: number, count: number, seed: number): number[] {
  const random = randomSource(seed);
  return Array.from({ length: count }, () => logGamma(alpha, random));
}

function softmax(logits: number[], legal: boolean[]): number[] {
  if (!legal.some(Boolean)) return logits.map(() => 1 / logits.length);
  const maximum = Math.max(...logits.filter((_, index) => legal[index]));
  const weights = logits.map((value, index) => legal[index] ? Math.exp(value - maximum) : 0);
  const total = weights.reduce((sum, value) => sum + value, 0);
  return weights.map(value => value / total);
}

/** Same masking and probability mixture as prepare_logits, given log-Gamma draws. */
export function policyDistributions(logits: number[], legal: boolean[], logNoise: number[], fraction: number): {
  raw: number[]; masked: number[]; noise: number[]; mixed: number[];
} {
  const raw = softmax(logits, logits.map(() => true));
  const masked = softmax(logits, legal);
  const noise = softmax(logNoise, legal);
  if (fraction === 0) return { raw, masked, noise, mixed: masked };
  const float32Tiny = 2 ** -126;
  const mixedLogits = masked.map((probability, index) =>
    Math.log(Math.max((1 - fraction) * probability + fraction * noise[index], float32Tiny)));
  const mixed = softmax(mixedLogits, legal);
  return { raw, masked, noise, mixed };
}
