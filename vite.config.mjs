import { resolve } from "node:path";
import { defineConfig } from "vite";

const root = resolve("frontend");

export default defineConfig({
  root,
  base: "/static/",
  build: {
    assetsDir: "assets",
    emptyOutDir: false,
    outDir: resolve("static"),
    rollupOptions: {
      input: {
        index: resolve(root, "index.html"),
        agents: resolve(root, "agents.html"),
        trade: resolve(root, "trade.html")
      }
    }
  }
});
