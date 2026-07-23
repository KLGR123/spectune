# Recommended `main` branch protection

Repository administrators should configure a GitHub ruleset for `main` with:

- require a pull request before merging;
- require at least one approval;
- dismiss stale approvals after new commits;
- require review from Code Owners;
- require all conversations to be resolved;
- require the `quality` and `metadata` status checks;
- block force pushes and branch deletion;
- apply the rules to administrators.

These settings enforce the review policy. `CODEOWNERS`, the pull request
template, and GitHub Actions support the policy but cannot replace branch
protection.
