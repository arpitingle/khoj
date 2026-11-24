import { writeFile } from 'node:fs/promises';

const apiUrl = (process.env.INDGOV_API_URL || '').trim().replace(/\/+$/, '');
if (!apiUrl.startsWith('https://')) {
  throw new Error('Set INDGOV_API_URL to the backend HTTPS URL ending in /api');
}

await writeFile('runtime-config.js', `window.INDGOV_API_URL = ${JSON.stringify(apiUrl)};\n`);
