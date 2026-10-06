import { defineConfig } from "oxlint";

// Anti-slop fence for the TSX half. Vendored ruleset: see tools/anti-slop/VENDOR.txt.
// The Python half is checked by ../tools/anti_slop.py, which ports these ideas.
export default defineConfig({
  // Built-in categories stay off: this fence is anti-slop, not an unused-var sweep.
  categories: {
    correctness: "off",
    suspicious: "off",
    pedantic: "off",
    perf: "off",
    style: "off",
    restriction: "off",
    nursery: "off",
  },
  plugins: [],
  ignorePatterns: ["node_modules/**", ".next/**", "tools/anti-slop/**"],
  jsPlugins: [{ name: "anti-slop", specifier: "./tools/anti-slop/index.ts" }],
  rules: {
    "anti-slop/no-module-mocking": "error",
    "anti-slop/no-chained-type-assertions": "error",
    "anti-slop/no-widen-then-assert": "error",
    "anti-slop/no-reflect-get": "error",
    "anti-slop/no-reflect-apply": "error",
    "anti-slop/no-unknown-type-aliases": "error",
    "anti-slop/no-shape-in-symbol-names": "error",
    "anti-slop/no-object-parameters": "error",
    "anti-slop/no-conditional-empty-object-spread": "error",
    "anti-slop/no-unknown-parameters": "error",
    "anti-slop/no-unknown-returns": "error",
    "anti-slop/no-known-value-widening": "error",
    "anti-slop/no-unsafe-dictionary-type": "error",
    "anti-slop/require-safety-comment-for-type-assertion": "error",
    "anti-slop/no-runtime-typeof": "error",
  },
});
