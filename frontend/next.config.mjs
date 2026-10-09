/** @type {import('next').NextConfig} */
const nextConfig = {
  // Required by docker/Dockerfile.frontend — the runner stage copies
  // .next/standalone (a minimal, self-contained server bundle).
  output: "standalone",
  async rewrites() {
    const backend = process.env.INTERNAL_API_URL || "http://localhost:8000";
    return [
      {
        source: "/api/backend/:path*",
        destination: `${backend}/:path*`,
      },
    ];
  },
};

export default nextConfig;
