import { hydrate, mount } from 'svelte';
import '@fontsource-variable/inter';
import '@fontsource-variable/source-serif-4';
import 'katex/dist/katex.min.css';
import './style.css';
import App from './App.svelte';

const target = document.getElementById('app')!;
if (target.querySelector('main')) hydrate(App, { target });
else mount(App, { target });
document.documentElement.classList.add('js');
