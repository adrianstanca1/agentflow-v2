export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: { sans: ['Geist', 'Inter', 'system-ui', 'sans-serif'], mono: ['Geist Mono', 'JetBrains Mono', 'monospace'] },
      colors: {
        teal: { 400: '#2dd4bf', 500: '#14b8a6', 600: '#0d9488' }
      }
    }
  },
  plugins: []
}
