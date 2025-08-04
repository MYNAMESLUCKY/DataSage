/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/auth/:path*',
        destination: 'http://localhost:8001/api/v1/auth/:path*'
      },
      {
        source: '/api/:path*',
        destination: 'http://localhost:8001/api/v1/:path*'
      }
    ]
  }
}

module.exports = nextConfig