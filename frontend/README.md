# Foresea Frontend

`frontend/` is the source for browser pages. Vite builds these entries into
`static/`, which remains the runtime directory served by FastAPI and packaged
into the Cloud Run image.

Current entries:

- `index.html` -> `/` and `/watchlist`
- `trade.html` -> `/trade`
- `agents.html` -> `/agents`

Run `npm run frontend:build` after editing frontend files.
