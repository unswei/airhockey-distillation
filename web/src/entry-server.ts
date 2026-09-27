import { render } from 'svelte/server';
import App from './App.svelte';
export const renderArticle = () => render(App);
