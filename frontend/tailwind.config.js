/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: "#6d28d9",
        accent: "#22d3ee",
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
}
