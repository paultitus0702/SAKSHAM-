/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        industrial: {
          950: '#0a0d12',
          900: '#10151f',
          850: '#161d2a',
          800: '#1e2838',
          700: '#2d3b52',
          600: '#425575',
          accent: '#f59e0b', // Amber 500 - industrial hazard/alert
          sovereign: '#06b6d4', // Cyan 500 - tech/sovereign
          success: '#10b981', // Emerald 500 - optimal
          warning: '#f97316', // Orange 500 - health warning
          danger: '#ef4444', // Red 500 - infeasible
        }
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', '"Liberation Mono"', '"Courier New"', 'monospace']
      }
    },
  },
  plugins: [],
}
