# Site

Personal site for articles and interactive visualisations.

## Stack

- **Astro + MDX:** website structure and article content.
- **React:** interactive components, state, and controls.
- **D3.js:** visualisation geometry, scales, and SVG paths.
- **Motion:** animated transitions and interactions.
- **KaTeX:** inline and display formulas.

Motion is part of the agreed stack but is not installed yet. The wave demo uses
D3's timer for continuous animation. Keep one animation owner per property.

## Development

Requires Node.js 24.22.2 or newer.

From this directory:

```sh
npm ci
npm run dev
```

Open the local URL printed by Astro. Edit `src/pages/index.mdx` for the homepage;
add `.mdx` files under `src/pages/` for more pages. Shared markup and styles live
in `src/layouts/ArticleLayout.astro`.

Use `$...$` for inline math and `$$` blocks for display equations. The example
figure is `src/components/WaveDemo.tsx`; SVG images live in `public/images/`.
D3 supplies scales, SVG path generation, and animation timing; React renders
the figure and handles the controls.

```sh
npm run build
npm run preview
```

The production build is written to `dist/`.
