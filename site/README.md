# Site

Astro + MDX, built as a static site. Requires Node.js 24.22.2 or newer.

From this directory:

```sh
npm ci
npm run dev
```

Open the local URL printed by Astro. Edit `src/pages/index.mdx` for the homepage;
add `.mdx` files under `src/pages/` for more pages. Shared markup and styles live
in `src/layouts/ArticleLayout.astro`.

```sh
npm run build
npm run preview
```

The production build is written to `dist/`.
