export default {
  extends: ["@commitlint/config-conventional"],
  parserPreset: {
    parserOpts: {
      issueNumber: ["#"],
    },
  },
  rules: {
    "references-empty": [2, "never"],
  },
};
