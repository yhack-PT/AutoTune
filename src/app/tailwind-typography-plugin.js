/* eslint-disable @typescript-eslint/no-require-imports */

// Next 16's Turbopack CSS pipeline can fail to resolve package names in
// Tailwind's `@plugin` directive even when Node can resolve them normally.
module.exports = require("@tailwindcss/typography");
