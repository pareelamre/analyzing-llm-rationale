# Multi-stage build → tiny static image.
FROM golang:1.22-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY *.go ./
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/marketd .

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/marketd /marketd
EXPOSE 8090
ENV PORT=8090
USER nonroot:nonroot
ENTRYPOINT ["/marketd"]
