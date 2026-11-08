# Site guidelines

- Apply these guidelines throughout `site/`, alongside the repository guidelines.
- Keep pages visually consistent when adding or changing content.

## Visual style

- Preserve the quiet article style: warm off-white backgrounds, dark text, serif prose, system-ui controls and captions, restrained borders, rounded figure panels, and generous spacing.
- Keep typography, content width, spacing, panel treatments, and control states consistent across pages. Use the existing muted indigo for primary controls and orange for highlights; use additional colors only when they clarify the visualization.
- Give equivalent concepts and interaction states consistent colors and labels. Explain chart-specific encodings with labels or legends; do not rely on color alone.
- Keep diagrams responsive and labels readable on narrow screens. Contain necessary scrolling within wide diagrams or tables instead of overflowing the page.

## Interactive visualizations and animations

- Use consistent control labels, ordering, sizing, and behavior for play/pause, stepping, reset, speed, and sliders wherever applicable. Show current values and units, and make selected, disabled, hover, and keyboard-focus states clear.
- Start animations paused and let the reader initiate playback. Respect `prefers-reduced-motion`, retaining a usable static or manually stepped explanation.
- Use motion to explain changes and relationships. Keep transition timing and easing consistent across related figures, and avoid decorative motion or layout shifts.
- Pause animation work while offscreen or while the document is hidden. Clean up timers, observers, and event listeners when components unmount.
- Use semantic controls with accessible labels, keyboard operation, and visible focus indicators. Give figures meaningful captions and diagrams accessible descriptions; keep the explanation understandable without playback.

## Reuse and verification

- Keep prose and page composition in MDX, shared article structure in Astro layouts, and interactive behavior in React components. Reuse existing D3 dependencies where useful.
- Extract reusable components when they remove meaningful duplication or keep shared behavior consistent, such as figure panels, playback controls, labeled sliders, and legends. Keep APIs small and domain-specific simulation logic local to each visualization.
- Centralize repeated visual values in shared CSS custom properties and reusable styles as they arise. Scope visualization-specific CSS to its component; update shared styles instead of copying slightly different versions into new pages.
- Reuse animation lifecycle logic when multiple figures need the same behavior. Avoid building a general visualization framework for isolated examples.
- For UI changes, check affected pages at desktop and narrow widths, keyboard interaction, reduced motion, playback controls, and visibility handling. When changing shared components or styles, check every consuming page for visual consistency and regressions.
- Run `npm run build` from `site/` for site code changes. Add focused behavior tests for changed interactive logic where practical, and report any checks that could not be performed. Documentation-only changes do not require a build.
