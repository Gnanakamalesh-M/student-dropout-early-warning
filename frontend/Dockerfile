# Dashboard image: build with Node, serve the static bundle with nginx.
#
# Built and run locally as well as in CI. The image built first try; its
# healthcheck did not — see the note on the HEALTHCHECK line below.

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
    # 127.0.0.1, not localhost: the container's /etc/hosts maps localhost to both
    # 127.0.0.1 and ::1, busybox wget tries ::1 first, and `listen 8080` binds
    # IPv4 only -- so the healthcheck failed with "Connection refused" and marked
    # the container unhealthy while it was serving HTTP 200 to the host the whole
    # time. The alternative fix is `listen [::]:8080` in nginx.conf; addressing
    # the loopback explicitly is narrower than changing what the server binds.
    CMD wget -q --spider http://127.0.0.1:8080/ || exit 1
