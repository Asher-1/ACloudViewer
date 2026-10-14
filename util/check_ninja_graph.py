# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Check a freshly configured Ninja build tree for plan-time errors.

Ninja validates the requested part of the graph before it runs a single
command and aborts on the first problem it finds:

  * an input that does not exist and that no build statement produces
    ("'<path>', needed by '<target>', missing and no known rule to make it"),
  * a dependency cycle,
  * an output produced by more than one build statement (dupbuild=err; this
    one is rejected while parsing, anywhere in the manifest).

Make never performs these checks, so a project can build cleanly with Make
while a Ninja build of the same configuration fails immediately. This script
parses the generated build files, walks the closure of the default targets
plus every --target, and reports every such problem at once.

Run it right after configuring a fresh build directory, before anything is
built: a missing input only goes unnoticed by Ninja when an earlier build
happened to leave the file behind.

Multi-config trees (build-<Config>.ninja) are checked per configuration;
single-config trees use build.ninja.

Usage: check_ninja_graph.py <build_dir> [--config NAME ...] [--target T ...]
"""

import argparse
import os
import sys


class NinjaSyntaxError(Exception):
    pass


def _logical_lines(path):
    """Yield (first physical line number, line) with '$'-newlines joined."""
    with open(path, encoding="utf-8", errors="surrogateescape") as f:
        pending, start = None, 0
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\r\n")
            if pending is not None:
                line = pending + line.lstrip(" ")
            else:
                start = lineno
            # An odd number of trailing '$' is an escaped newline.
            trailing = len(line) - len(line.rstrip("$"))
            if trailing % 2 == 1:
                pending = line[:-1]
                continue
            pending = None
            yield start, line
        if pending is not None:
            yield start, pending


def _expand(text, scope):
    """Evaluate the escapes and variable references of a Ninja string."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c != "$":
            out.append(c)
            i += 1
            continue
        if i + 1 >= n:
            raise NinjaSyntaxError("dangling '$' in: " + text)
        nxt = text[i + 1]
        if nxt in " :$":
            out.append(nxt)
            i += 2
        elif nxt == "{":
            end = text.index("}", i + 2)
            out.append(scope.get(text[i + 2:end], ""))
            i = end + 1
        else:
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] in "_-"):
                j += 1
            out.append(scope.get(text[i + 1:j], ""))
            i = j
    return "".join(out)


def _raw_tokens(text):
    """Split at spaces and the first ':' that are not escaped by '$'.

    Returns (tokens, index of the first token after that ':', i.e. the rule).
    """
    tokens, cur, colon, i = [], [], None, 0
    while i < len(text):
        c = text[i]
        if c == "$":
            cur.append(text[i:i + 2])
            i += 2
            continue
        if c == " " or (c == ":" and colon is None):
            if cur:
                tokens.append("".join(cur))
                cur = []
            if c == ":":
                colon = len(tokens)
        else:
            cur.append(c)
        i += 1
    if cur:
        tokens.append("".join(cur))
    return tokens, colon


# Separators of the implicit, order-only and validation groups. Ninja requires
# every group to exist or be produced, so the groups need no distinction here.
_GROUP_MARKERS = {"|", "||", "|@"}


class Graph:

    def __init__(self, build_dir):
        self.build_dir = build_dir
        self.producers = {}  # output -> location of its build statement
        self.deps = {}  # output -> inputs of its build statement
        self.defaults = []
        self.statements = 0
        self.duplicates = []  # (output, first location, second location)

    def path(self, token):
        return os.path.normpath(os.path.join(self.build_dir, token))

    def load(self, ninja_file, scope=None):
        scope = {} if scope is None else scope
        path = os.path.join(self.build_dir, ninja_file)
        for lineno, line in _logical_lines(path):
            if not line or line.startswith("#") or line.startswith(" "):
                # Indented bindings belong to the preceding statement; Ninja
                # never resolves inputs/outputs from them.
                continue
            location = "{}:{}".format(ninja_file, lineno)
            keyword, _, rest = line.partition(" ")
            if keyword == "build":
                self._add_build(rest, scope, location)
            elif keyword == "include":
                self.load(_expand(rest.strip(), scope), scope)
            elif keyword == "subninja":
                self.load(_expand(rest.strip(), scope), dict(scope))
            elif keyword == "default":
                tokens, _ = _raw_tokens(rest)
                self.defaults += [self.path(_expand(t, scope)) for t in tokens]
            elif keyword in ("rule", "pool"):
                continue
            else:
                name, eq, value = line.partition("=")
                if not eq:
                    raise NinjaSyntaxError("{}: unexpected line: {}".format(
                        location, line))
                scope[name.strip()] = _expand(value.lstrip(), scope)

    def _add_build(self, rest, scope, location):
        tokens, colon = _raw_tokens(rest)
        if colon is None or colon == len(tokens):
            raise NinjaSyntaxError(location + ": malformed build line")
        # tokens[colon] is the rule name.
        outputs = [
            self.path(_expand(t, scope))
            for t in tokens[:colon]
            if t not in _GROUP_MARKERS
        ]
        inputs = [
            self.path(_expand(t, scope))
            for t in tokens[colon + 1:]
            if t not in _GROUP_MARKERS
        ]
        self.statements += 1
        for out in outputs:
            first = self.producers.setdefault(out, location)
            if first != location:
                self.duplicates.append((out, first, location))
            self.deps[out] = inputs

    def plan(self, roots):
        """Walk the closure of `roots` the way Ninja plans a build.

        Returns ({missing input: a consumer}, one cycle as a path list or
        None). Statements outside the closure are never checked by Ninja: a
        multi-config tree carries utility statements of every configuration
        in its shared common.ninja.
        """
        missing, cycle = {}, None
        # Iterative DFS, O(V + E): the graph has ~1e5 nodes, deeper than the
        # recursion limit allows.
        state = {}  # 1 = on the current path, 2 = finished
        for root in roots:
            if root in state:
                continue
            state[root] = 1
            stack = [(root, iter(self.deps.get(root, ())))]
            while stack:
                node, children = stack[-1]
                child = next(children, None)
                if child is None:
                    state[node] = 2
                    stack.pop()
                elif state.get(child) == 1:
                    if cycle is None:
                        path = [n for n, _ in stack]
                        cycle = path[path.index(child):] + [child]
                elif child in state:
                    continue
                elif child in self.deps:
                    state[child] = 1
                    stack.append((child, iter(self.deps[child])))
                else:
                    state[child] = 2
                    if not os.path.exists(child):
                        missing[child] = node
        return missing, cycle


def _build_files(build_dir, configs):
    if configs:
        return ["build-{}.ninja".format(c) for c in configs]
    per_config = sorted(f for f in os.listdir(build_dir)
                        if f.startswith("build-") and f.endswith(".ninja"))
    # build.ninja of a multi-config tree includes every build-<Config>.ninja;
    # Ninja only ever plans one configuration at a time.
    return per_config or ["build.ninja"]


def check(build_dir, ninja_file, targets):
    graph = Graph(build_dir)
    graph.load(ninja_file)
    problems = 0
    roots = list(graph.defaults)
    for target in targets:
        root = graph.path(target)
        if root not in graph.deps:
            print("{}: unknown target '{}'".format(ninja_file, target))
            problems += 1
        else:
            roots.append(root)
    if not roots:
        print("{}: no default statement and no --target".format(ninja_file))
        return problems + 1
    missing, cycle = graph.plan(roots)
    for path, consumer in sorted(missing.items()):
        print("{}: missing and no known rule to make it: {}\n    needed by {}".
              format(ninja_file, path, consumer))
    problems += len(missing)
    # Ninja rejects duplicate outputs while parsing, wherever they are.
    for out, first, second in graph.duplicates:
        print("{}: multiple rules generate {} ({}, {})".format(
            ninja_file, out, first, second))
    problems += len(graph.duplicates)
    if cycle:
        print("{}: dependency cycle: {}".format(ninja_file, " -> ".join(cycle)))
        problems += 1
    print("{}: {} build statements, {} planned roots, {} problem(s)".format(
        ninja_file, graph.statements, len(roots), problems))
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("build_dir", help="freshly configured Ninja tree")
    parser.add_argument("--config",
                        action="append",
                        default=[],
                        help="configuration to check (default: all)")
    parser.add_argument("--target",
                        action="append",
                        default=[],
                        help="target built besides the default ones")
    args = parser.parse_args()
    build_dir = os.path.abspath(args.build_dir)
    problems = 0
    for ninja_file in _build_files(build_dir, args.config):
        problems += check(build_dir, ninja_file, args.target)
    if problems:
        print("Ninja build graph check FAILED: {} problem(s)".format(problems))
        return 1
    print("Ninja build graph check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
