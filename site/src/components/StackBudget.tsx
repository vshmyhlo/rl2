import { useState } from 'react';
import './Stack.css';

const scenarios = [
  {
    label: 'One default job',
    jobs: [{ name: 'A', gpu: 1, cpu: 10, ram: 32 }],
    note: 'One GPU is free, but all 10 CPU cores are reserved. A second default job waits.',
  },
  {
    label: 'Two smaller jobs',
    jobs: [
      { name: 'A', gpu: 1, cpu: 5, ram: 32 },
      { name: 'B', gpu: 1, cpu: 5, ram: 32 },
    ],
    note: 'With 5 CPU cores requested per job, both one-GPU jobs fit exactly within the queue budget.',
  },
  {
    label: 'One two-GPU job',
    jobs: [{ name: 'A', gpu: 2, cpu: 10, ram: 32 }],
    note: 'One job reserves both GPUs and all CPU quota. Spare RAM alone cannot admit another GPU job.',
  },
];

const resources = [
  { key: 'gpu', label: 'GPUs', quota: 2, unit: 'GPUs' },
  { key: 'cpu', label: 'CPU', quota: 10, unit: 'cores' },
  { key: 'ram', label: 'RAM', quota: 64, unit: 'GiB' },
] as const;

export default function StackBudget() {
  const [selected, setSelected] = useState(0);
  const scenario = scenarios[selected];

  return (
    <figure className="stack-figure" aria-labelledby="stack-budget-caption">
      <div className="stack-panel">
        <p className="stack-label">Explore the queue budget</p>
        <div className="stack-controls" role="group" aria-label="Job allocation scenario">
          {scenarios.map((item, index) => (
            <button
              key={item.label}
              type="button"
              aria-pressed={selected === index}
              onClick={() => setSelected(index)}
            >
              {item.label}
            </button>
          ))}
        </div>
        <div aria-live="polite" aria-atomic="true">
          {resources.map(({ key, label, quota, unit }) => {
            const reserved = scenario.jobs.reduce((sum, job) => sum + job[key], 0);
            return (
              <div className="stack-resource" key={key}>
                <div className="stack-resource-heading">
                  <strong>{label}</strong>
                  <span>{reserved} / {quota} {unit} reserved</span>
                </div>
                <div
                  className="stack-bar"
                  role="img"
                  aria-label={`${label}: ${scenario.jobs.map((job) => `Job ${job.name} reserves ${job[key]} ${unit}`).join('; ')}; ${quota - reserved} ${unit} free`}
                >
                  {scenario.jobs.map((job) => (
                    <span
                      key={job.name}
                      className={`stack-job stack-job-${job.name.toLowerCase()}`}
                      style={{ width: `${job[key] / quota * 100}%` }}
                    >
                      {job.name} · {job[key]}
                    </span>
                  ))}
                  {reserved < quota && <span className="stack-free">Free · {quota - reserved}</span>}
                </div>
              </div>
            );
          })}
          <p className="stack-budget-note">{scenario.note}</p>
        </div>
      </div>
      <figcaption id="stack-budget-caption">A and B are example jobs. Bars show requested resources against each resource’s quota, not measured utilization.</figcaption>
    </figure>
  );
}
