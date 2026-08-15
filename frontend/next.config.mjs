/** @type {import('next').NextConfig} */
const nextConfig = {
  // Required by docker/Dockerfile.frontend — the runner stage copies
  // .next/standalone (a minimal, self-contained server bundle).
  output: "standalone",
};

export default nextConfig;
