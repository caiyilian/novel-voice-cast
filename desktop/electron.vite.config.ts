import tailwindcss from "@tailwindcss/vite"
import { defineConfig } from "electron-vite"
import solid from "vite-plugin-solid"

export default defineConfig({
  main: {
    build: {
      rollupOptions: {
        input: "src/main/index.ts",
      },
    },
  },
  preload: {
    build: {
      rollupOptions: {
        input: "src/preload/index.ts",
        output: {
          format: "cjs",
          entryFileNames: "index.js",
        },
      },
    },
  },
  renderer: {
    root: "src/renderer",
    plugins: [solid(), tailwindcss()],
    build: {
      rollupOptions: {
        input: "src/renderer/index.html",
      },
    },
  },
})
