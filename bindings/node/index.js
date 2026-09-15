'use strict'
// Local/dev loader for the rwkv-router native binding.
//
// The npm publish flow (@napi-rs/cli) ships one `*.node` per platform
// package; for local development a single conventional artifact next to this
// file is enough (see README "Build from source"):
//
//   cargo build --release -p rwkv-router-node
//   copy ../../target/release/rwkv_router_node.dll -> rwkv-router.win32-x64-msvc.node

const fs = require('node:fs')
const path = require('node:path')

function locate() {
  const dir = __dirname
  // 1. Conventional per-platform artifacts (napi-rs naming).
  const conventional = [
    'rwkv-router.win32-x64-msvc.node',
    'rwkv-router.win32-arm64-msvc.node',
    'rwkv-router.darwin-arm64.node',
    'rwkv-router.darwin-x64.node',
    'rwkv-router.linux-x64-gnu.node',
    'rwkv-router.linux-arm64-gnu.node',
  ].find((f) => fs.existsSync(path.join(dir, f)))
  if (conventional) return require(path.join(dir, conventional))
  // 2. Any locally built *.node fallback.
  const any = fs.readdirSync(dir).find((f) => f.endsWith('.node'))
  if (any) return require(path.join(dir, any))
  throw new Error(
    'rwkv-router: native binding not found next to index.js. ' +
      'Build from source: `cargo build --release -p rwkv-router-node`, then copy ' +
      '../../target/release/rwkv_router_node(.dll/.so/.dylib) here as ' +
      'rwkv-router.<platform>.node (e.g. rwkv-router.win32-x64-msvc.node).'
  )
}

module.exports = locate()
