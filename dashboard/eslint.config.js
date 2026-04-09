#!/usr/bin/env node
/**
 * eslint.config.js --- flat ESLint configuration for the dashboard
 *
 * Contains:
 *   default export: lint rules for the TypeScript/React source tree
 */

import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist"] },
  ...tseslint.configs.recommended,
  {
    rules: {
      "@typescript-eslint/no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
    },
  },
);
