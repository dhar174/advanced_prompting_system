// tailwind.config.js
module.exports = {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        'brand-primary': '#2563EB',
        'brand-secondary': '#4B5563',
        'surface-ground': '#F3F4F6', // Adjusted to a common v3 slate/gray
        'surface-card': '#FFFFFF',
        'text-primary': '#1E293B',
        'text-secondary': '#475569',
        'assistant-bubble-bg': '#E2E8F0',
        'user-bubble-bg': '#2563EB',
        'user-bubble-text': '#FFFFFF',
        // ring-brand was defined before, ensure it's compatible or use Tailwind's default focus rings
        // For v3, focus ring utilities are more common (e.g., ring-blue-500)
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', "Segoe UI", 'Roboto', "Helvetica Neue", 'Arial', "Noto Sans", 'sans-serif', "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"], // Adjusted to match previous full list
      },
    },
  },
  plugins: [
    // require('tailwind-scrollbar'), // Keep this commented out for now
  ],
};
