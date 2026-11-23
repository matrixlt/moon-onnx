#!/bin/sh
export PATH="$HOME/.moon/bin:$PATH"
exec moonx moonbitlang/protoc-gen-mbt@0.3.0 "$@"
