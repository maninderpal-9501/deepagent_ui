import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Required for the multi-stage Docker build: generates a self-contained
  // .next/standalone directory with a minimal node_modules subset and a
  // server.js entrypoint — no full node_modules needed in the runner image.
  output: "standalone",
};

export default nextConfig;
