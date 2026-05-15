FROM node:20-alpine AS build

WORKDIR /app

COPY dashboard/package.json ./
RUN npm install --no-audit --no-fund

COPY dashboard/ ./
RUN npm run build

FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK CMD wget -q -O /dev/null http://localhost/ || exit 1
