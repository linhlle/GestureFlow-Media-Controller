// Emit configs with the builder's own serializer so Python can try to load them.
//
// tests/test_web_parity.py compares schema.js constants against commands.py
// textually. That catches a threshold drifting apart; it does not catch the
// serializer emitting something no YAML parser accepts, which is how the stock
// "zoom in" binding shipped as `keys: [command, =]` -- valid-looking, and
// unloadable.
//
// Usage: node scripts/config_check.mjs <fixtures.json>

import { readFileSync } from 'node:fs';
import { toYaml } from '../web/js/schema.js';

const fixtures = JSON.parse(readFileSync(process.argv[2], 'utf8'));
process.stdout.write(JSON.stringify(fixtures.map((cfg) => toYaml(cfg))));
