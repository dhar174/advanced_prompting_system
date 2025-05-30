/** @type {import('tailwindcss').Config} */
import colors from 'tailwindcss/colors';

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'brand-primary': colors.blue[600], // #2563EB
        'brand-secondary': colors.gray[600], // #4B5563
        'surface-ground': colors.slate[100], // #F1F5F9
        'surface-card': colors.white,       // #FFFFFF
        'text-primary': colors.slate[900],   // #0F172A
        'text-secondary': colors.slate[600], // #475569
        'assistant-bubble-bg': colors.slate[200], // #E2E8F0
        'user-bubble-bg': colors.blue[600], // #2563EB (brand-primary)
        'user-bubble-text': colors.white,   // #FFFFFF
        // Adding ring/focus colors for consistency
        'ring-brand': colors.blue[500], // For focus rings
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', "Segoe UI", 'Roboto', "Helvetica Neue", 'Arial', "Noto Sans", 'sans-serif', "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"],
      },
    },
  },
  plugins: [
    require('tailwind-scrollbar'),
  ],
}
