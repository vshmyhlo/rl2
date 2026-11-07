import { useEffect, useMemo, useRef, useState, type JSX } from 'react';
import { scaleLinear } from 'd3-scale';
import { line } from 'd3-shape';
import { timer } from 'd3-timer';
import './WaveDemo.css';

const fullTurn = 2 * Math.PI;
const phases = Array.from({ length: 129 }, (_, index) => (index / 128) * fullTurn);
const waveScale = scaleLinear().domain([0, fullTurn]).range([265, 585]);
const orbitScale = scaleLinear().domain([-1, 1]).range([50, 180]);
const heightScale = scaleLinear().domain([-1, 1]).range([205, 75]);

export default function WaveDemo(): JSX.Element {
  const [amplitude, setAmplitude] = useState(1);
  const [angle, setAngle] = useState(Math.PI / 4);
  const [playing, setPlaying] = useState(false);
  const [visible, setVisible] = useState(true);
  const [pageVisible, setPageVisible] = useState(true);
  const container = useRef<HTMLElement>(null);
  const currentAngle = useRef(Math.PI / 4);

  const radius = heightScale(0) - heightScale(amplitude);
  const height = heightScale(amplitude * Math.sin(angle));
  const orbitX = orbitScale(amplitude * Math.cos(angle));
  const waveX = waveScale(angle);
  const wave = useMemo(() => (
    line<number>()
      .x((phase) => waveScale(phase))
      .y((phase) => heightScale(amplitude * Math.sin(phase)))(phases)
  ), [amplitude]);

  useEffect(() => {
    const element = container.current;
    if (!element) return;
    const observer = new IntersectionObserver(([entry]) => {
      setVisible(entry.isIntersecting);
    });
    observer.observe(element);
    const updateVisibility = (): void => {
      setPageVisible(!document.hidden);
    };
    updateVisibility();
    document.addEventListener('visibilitychange', updateVisibility);
    return () => {
      observer.disconnect();
      document.removeEventListener('visibilitychange', updateVisibility);
    };
  }, []);

  useEffect(() => {
    if (!playing || !visible || !pageVisible) return;
    // Keep the resume position without restarting the timer on every frame.
    const startAngle = currentAngle.current;
    const animation = timer((elapsed) => {
      currentAngle.current = (startAngle + (elapsed / 6000) * fullTurn) % fullTurn;
      setAngle(currentAngle.current);
    });
    return () => animation.stop();
  }, [playing, visible, pageVisible]);

  return (
    <figure ref={container} className="wave-demo">
      <svg viewBox="0 0 620 260" role="img" aria-label="Linked circle and sine wave. Orange dots mark the same height on both diagrams.">
        <g className="axes">
          <path d="M20 140H215M115 40V240M265 140H595M265 40V240" />
          <circle cx="115" cy="140" r={radius} />
        </g>
        <path d={wave ?? undefined} className="curve" />
        <path d={`M${orbitX} ${height}H${waveX}`} className="guide" />
        <path d={`M115 140L${orbitX} ${height}`} className="curve" />
        <circle cx={orbitX} cy={height} r="6" className="dot" />
        <circle cx={waveX} cy={height} r="6" className="dot" />
        <g className="labels">
          <text x="115" y="24" textAnchor="middle">Around a circle</text>
          <text x="425" y="24" textAnchor="middle">Height over one turn</text>
          <text x="265" y="250" textAnchor="middle">0</text>
          <text x="425" y="250" textAnchor="middle">π</text>
          <text x="585" y="250" textAnchor="middle">2π</text>
        </g>
      </svg>
      <div className="controls">
        <button type="button" onClick={() => setPlaying((value) => !value)} aria-pressed={playing}>
          {playing ? 'Pause' : 'Play'}
        </button>
        <label>
          <span>Angle <output>{(angle / Math.PI).toFixed(2)}π</output></span>
          <input
            aria-label="Angle"
            type="range"
            min="0"
            max={fullTurn}
            step="0.01"
            value={angle}
            onChange={(event) => {
              setPlaying(false);
              currentAngle.current = event.currentTarget.valueAsNumber;
              setAngle(currentAngle.current);
            }}
          />
        </label>
        <label>
          <span>Amplitude <output>{amplitude.toFixed(2)}</output></span>
          <input
            aria-label="Amplitude"
            type="range"
            min="0"
            max="1.4"
            step="0.05"
            value={amplitude}
            onChange={(event) => setAmplitude(event.currentTarget.valueAsNumber)}
          />
        </label>
      </div>
      <figcaption>Fig. 2 — Two views of the same motion. Animation starts only when you press Play.</figcaption>
    </figure>
  );
}
