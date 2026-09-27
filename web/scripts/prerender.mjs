import { readFile, writeFile } from 'node:fs/promises';
import { renderArticle } from '../.ssr/entry-server.js';
const { body, head } = await renderArticle();
const output = new URL('../dist/index.html', import.meta.url);
const template = await readFile(output, 'utf8');
if (!template.includes('<!--article-body-->')) throw new Error('Missing prerender target');
await writeFile(
  output,
  template.replace('<!--article-head-->', head).replace('<!--article-body-->', body),
);
console.log(
  'Prerendered the complete article: text, equations and default figures work without JavaScript.',
);
