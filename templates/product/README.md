# __name__

__name__ in the browser, on the CPU, compiled to WebAssembly with
[fusor-ai](https://github.com/fusor-rs/fusor-ai). Once the model has loaded, the page makes no
network requests.

## Build

You need Rust 1.99 or newer, Node.js 22 or newer and [`just`](https://github.com/casey/just).
Clone with `--recurse-submodules`, then:

```sh
just build    # the threads toolchain and tools, the CLI, the bundle, the runtimes and the page
just serve    # the page on http://localhost:8787/
```

`just check` lints the page and the scripts; `just smoke` and `just e2e` run the runtimes in Node
and in headless Chrome.
