# AGENTS.md

Rules for coding agents in this repository that are its own. fusor-ai's
`vendor/fusor-ai/AGENTS.md` is the standard and binds here too (`CLAUDE.md`
imports both); this file adds what is particular to this product.

__name__ runs its model in the browser on the CPU, sealed from the network. The
README describes what it does; don't invent scope, features or options beyond
the task.

## Repository

- `vendor/fusor-ai/models/__name__`: the model, its compiler, its runtime and
  its CLI, in fusor-ai.
- `app/`: the page, a [fusor](https://github.com/fusor-rs/fusor) app with its own
  Cargo workspace: its state as Rust signals in `src/app.rs`, the template in
  `web/index.html` on fusor-ai's `packages/ui/base.css`, the engine bridge in
  `web/app.js`, and in `engine/` the engine worker on fusor-ai's worker kit,
  beside the runtimes `just runtimes` builds.
- `scripts/node-smoke.mjs`: the Node smoke test; `e2e/run.mjs`: the headless
  Chrome run on fusor-ai's e2e kit.

What belongs where: model code and anything another model could reuse go to
fusor-ai, following its `AGENTS.md`. This repository holds what turns the model
into an app: the page, the checks against the reference and the deployment. A
change to fusor-ai lands there first; this repository never patches it.

Constraints that hold everywhere:

- fusor-ai comes only from `vendor/fusor-ai`, never a crates.io version, a
  `[patch]` table or a sibling checkout. The submodule moves to another pushed
  fusor-ai commit in a change of its own. No copies of fusor-ai code here.
- The page owns every capability: caching the bundle (OPFS) and measuring
  itself. The runtime does no I/O.

## Checks

A change is done only when this passes:

```sh
just check
```

It runs formatting and Clippy on the page and syntax checks of the scripts;
fusor-ai's `just check` covers the runtime and the CLI. A change to the page or
fusor-ai also needs `just runtimes`, `just smoke` and `just e2e`.
