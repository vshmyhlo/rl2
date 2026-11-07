# Site

Astro + MDX, with KaTeX formulas and a React + D3 figure. Requires Node.js
24.22.2 or newer.

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
