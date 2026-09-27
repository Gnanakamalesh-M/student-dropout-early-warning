# Dashboard image: build with Node, serve the static bundle with nginx.
#
# NOT BUILT OR RUN BY ITS AUTHOR — see the note in the root Dockerfile. The
# `containers` CI job is what verifies this.

FROM node:22-bookworm-slim AS builder

WORKDIR /app
# package-lock.json first so `npm ci` caches independently of source changes.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
# `npm run build` runs tsc then vite build, so a type error fails the image
# rather than shipping. Phase 10 found a Plotly trace missing from the basic
# distribution this way.
RUN npm run build


FROM nginx:1.27-alpine AS runtime

# nginx:alpine already runs workers as an unprivileged user; the config below
# binds 8080 so the master does not need to either.
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q --spider http://localhost:8080/ || exit 1
