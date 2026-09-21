# AISHA Studio

Local React/TypeScript/Vite development UI for AISHA Core. AISHA runs entirely on
your machine: Studio does not call hosted APIs or own conversation memory.

## Start

Start AISHA Core on 127.0.0.1:8000 first, then:

```bash
cd apps/aisha-studio
npm install
npm run dev
```

Open http://127.0.0.1:5173. Requires Node.js 20.19+.

Vite proxies HTTP and WebSocket `/v1` traffic to Core. Do not expose
port 8000 or 5173 to the public network: this early development build
has no user authentication.

`npm run build` checks TypeScript and builds `dist/`. Production hosting,
voice input/output, memory editing, and the avatar are later milestones.
