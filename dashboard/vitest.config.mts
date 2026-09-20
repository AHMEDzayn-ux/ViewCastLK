import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  test: {
    environment: "node",
    // Windows workers intermittently exceed the default teardown timeout when
    // the files run in parallel, which reports passing tests as failures. The
    // suite takes about a minute either way, so it runs serially.
    fileParallelism: false,
  },
});
